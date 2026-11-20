//! Secant probe state survives resume, selected resume, and refresh.

use super::fixture::{self, Client, Result, Snapshot};
use super::shapes::{self, Lcg, Row};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedMappedRootProgress,
    CertifiedRootAdvanceAllowance, CertifiedRootContinuation, CertifiedRootDomainQuery,
    CertifiedRootMappedValues, CertifiedRootProbe, CertifiedRootTolerance,
};

const COARSE: f64 = 0.05;
const ROWS: usize = 20;

fn fine(dtype: DType) -> f64 {
    if dtype == DType::F32 { 2e-3 } else { 2e-10 }
}

fn tolerance(atol: f64) -> CertifiedRootTolerance {
    CertifiedRootTolerance { atol, rtol: 0.0 }
}

fn allowance(max_iterations: usize) -> CertifiedRootAdvanceAllowance {
    CertifiedRootAdvanceAllowance {
        max_iterations,
        max_points: 1 << 20,
    }
}

/// Mapped coordinates are twice the parameter, an exact scaling.
fn doubled<R, C>(
    client: &C,
    query: &CertifiedRootDomainQuery<R>,
) -> numr::error::Result<CertifiedRootMappedValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    Ok(CertifiedRootMappedValues {
        lower: client.mul_scalar(&query.lower, 2.0)?,
        upper: client.mul_scalar(&query.upper, 2.0)?,
    })
}

struct Session<'a, R: Runtime<DType = DType>, C> {
    client: &'a C,
    rows: Vec<Row>,
    params: shapes::Params<R>,
    dtype: DType,
}

impl<'a, R, C> Session<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    fn new(client: &'a C, dtype: DType) -> Result<Self> {
        let mut random = Lcg(0xc0de + dtype.size_in_bytes() as u64);
        let rows: Vec<_> = (0..ROWS).map(|_| random.row(dtype)).collect();
        let params = shapes::params::<R>(&rows, dtype, client.device())?;
        Ok(Self {
            client,
            rows,
            params,
            dtype,
        })
    }

    fn begin(&self, atol: f64) -> Result<CertifiedMappedRootProgress<R>> {
        let column = |pick: fn(&Row) -> f64| {
            let values: Vec<_> = self.rows.iter().map(pick).collect();
            shapes::tensor::<R>(&values, self.dtype, self.client.device())
        };
        Ok(self.client.bisect_certified_mapped(
            |query| shapes::evaluate(self.client, &self.params, &query.owner, &query.point),
            |query| doubled(self.client, query),
            &column(|row| row.lower)?,
            &column(|row| row.upper)?,
            &column(|row| row.shape.kappa())?,
            &CertifiedBisectionOptions {
                atol,
                rtol: 0.0,
                max_iterations: 400,
                max_points: 1 << 20,
                probe: CertifiedRootProbe::SlackSecant,
            },
        )?)
    }

    /// Advance the given owners, or every owner, and check that no other owner is queried.
    fn advance(
        &self,
        continuation: CertifiedRootContinuation<R>,
        owners: Option<&[i64]>,
        refresh: bool,
        atol: f64,
        rounds: usize,
    ) -> Result<CertifiedMappedRootProgress<R>> {
        let guard = |queried: &Tensor<R>| -> numr::error::Result<()> {
            if let Some(owners) = owners {
                for owner in queried.try_to_vec::<i64>()? {
                    assert!(owners.contains(&owner), "undemanded owner {owner} queried");
                }
            }
            Ok(())
        };
        let residual = |query: &solvr::optimize::roots::CertifiedRootQuery<R>| {
            guard(&query.owner)?;
            shapes::evaluate(self.client, &self.params, &query.owner, &query.point)
        };
        let mapping = |query: &CertifiedRootDomainQuery<R>| {
            guard(&query.owner)?;
            doubled(self.client, query)
        };
        let (tolerance, allowance) = (tolerance(atol), allowance(rounds));
        let Some(owners) = owners else {
            return Ok(self.client.resume_certified_mapped(
                residual,
                mapping,
                continuation,
                &tolerance,
                &allowance,
            )?);
        };
        let owners = Tensor::<R>::from_slice(owners, &[owners.len()], self.client.device())?;
        Ok(if refresh {
            self.client.refresh_certified_mapped_selected(
                residual,
                mapping,
                continuation,
                &owners,
                &tolerance,
                &allowance,
            )?
        } else {
            self.client.resume_certified_mapped_selected(
                residual,
                mapping,
                continuation,
                &owners,
                &tolerance,
                &allowance,
            )?
        })
    }

    fn contains(&self, snapshot: &Snapshot) {
        for (index, &owner) in snapshot.owners.iter().enumerate() {
            let decode = |raw: u64| {
                if self.dtype == DType::F32 {
                    f64::from(f32::from_bits(raw as u32))
                } else {
                    f64::from_bits(raw)
                }
            };
            let (lo, hi) = (decode(snapshot.lower[index]), decode(snapshot.upper[index]));
            let (root_lower, root_upper) = self.rows[owner as usize].root(self.dtype);
            assert!(
                lo <= root_upper && hi >= root_lower,
                "owner {owner} excludes its root"
            );
        }
    }
}

/// Final enclosure bits per original owner.
fn per_owner(snapshot: &Snapshot) -> Vec<(i64, i64, u64, u64)> {
    snapshot
        .owners
        .iter()
        .enumerate()
        .map(|(index, &owner)| {
            (
                owner,
                snapshot.status[owner as usize],
                snapshot.lower[index],
                snapshot.upper[index],
            )
        })
        .collect()
}

/// Snapshots of every continuation path, in a fixed order.
pub fn trace<R, C>(client: &C, dtype: DType) -> Result<Vec<Snapshot>>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let session = Session::new(client, dtype)?;
    let fine = fine(dtype);
    let direct = fixture::snapshot(session.begin(fine)?.result())?;
    assert!(direct.status.iter().all(|&status| status == 0));
    session.contains(&direct);
    let mut snapshots = vec![direct];

    // Resume one round at a time from a coarse stop.
    let (coarse, mut continuation) = session.begin(COARSE)?.into_parts();
    let coarse = fixture::snapshot(&coarse)?;
    session.contains(&coarse);
    loop {
        let progress = session.advance(continuation, None, false, fine, 1)?;
        let snapshot = fixture::snapshot(progress.result())?;
        session.contains(&snapshot);
        let done = snapshot.status.iter().all(|&status| status == 0);
        continuation = progress.into_parts().1;
        if done {
            snapshots.push(snapshot);
            break;
        }
    }

    // Selected resume advances only the demanded owners.
    let demanded: Vec<i64> = (0..ROWS as i64).filter(|owner| owner % 3 != 1).collect();
    let (_, continuation) = session.begin(COARSE)?.into_parts();
    let selected = session.advance(continuation, Some(&demanded), false, fine, 400)?;
    snapshots.push(fixture::snapshot(selected.result())?);

    // Refresh keeps probe state, so the same residual continues along the same probes.
    let (_, continuation) = session.begin(COARSE)?.into_parts();
    let all: Vec<i64> = (0..ROWS as i64).collect();
    let refreshed = session.advance(continuation, Some(&all), true, fine, 400)?;
    snapshots.push(fixture::snapshot(refreshed.result())?);
    snapshots.push(coarse);
    Ok(snapshots)
}

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let snapshots = trace(client, dtype)?;
        let [direct, chunked, selected, refreshed, coarse] = &snapshots[..] else {
            unreachable!("trace returns five snapshots");
        };
        // Each row's probes depend only on its own state, so every path ends on the same bits.
        assert_eq!(per_owner(chunked), per_owner(direct), "{dtype:?} resume");
        assert_eq!(per_owner(refreshed), per_owner(direct), "{dtype:?} refresh");
        // Rows idle in different rounds after a coarse stop, but execute the direct points.
        assert!(chunked.iterations >= direct.iterations, "{dtype:?}");
        assert_eq!(chunked.points, direct.points, "{dtype:?}");
        for (index, &owner) in selected.owners.iter().enumerate() {
            let reference = if owner % 3 == 1 { coarse } else { direct };
            let position = reference
                .owners
                .iter()
                .position(|&other| other == owner)
                .unwrap_or(usize::MAX);
            assert_eq!(
                (
                    selected.status[owner as usize],
                    selected.lower[index],
                    selected.upper[index]
                ),
                (
                    reference.status[owner as usize],
                    reference.lower[position],
                    reference.upper[position]
                ),
                "{dtype:?} selected owner {owner}"
            );
        }
        let resumed = selected.points - coarse.points;
        let direct_extra = direct.points - coarse.points;
        assert!(resumed < direct_extra, "undemanded owners took points");
    }
    Ok(())
}
