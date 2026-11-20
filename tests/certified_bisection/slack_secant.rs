//! Slack-gated secant probes keep every certified guarantee of midpoint probes.

use super::callback;
use super::fixture::{self, Client, Result, decode};
use super::shapes::{self, Lcg, Row, Shape};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootQuery, CertifiedRootValues,
};
use std::cell::RefCell;
use std::collections::HashMap;

const SLACK: CertifiedRootProbe = CertifiedRootProbe::SlackSecant;
const MIDPOINT: CertifiedRootProbe = CertifiedRootProbe::Midpoint;

/// Host copy of one solve, including every callback query.
#[derive(Debug, PartialEq)]
pub struct Run {
    pub status: Vec<i64>,
    pub owner: Vec<i64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub iterations: usize,
    pub evaluated: usize,
    pub queries: Vec<(Vec<i64>, Vec<f64>)>,
    pub bits: Vec<u64>,
}

impl Run {
    /// Probe rounds each owner received after endpoint admission.
    fn rounds(&self, owner: i64) -> usize {
        self.queries
            .iter()
            .skip(2)
            .filter(|(owners, _)| owners.contains(&owner))
            .count()
    }
}

fn options(atol: f64, probe: CertifiedRootProbe) -> CertifiedBisectionOptions {
    CertifiedBisectionOptions {
        atol,
        rtol: 0.0,
        max_iterations: 400,
        max_points: 1 << 20,
        probe,
    }
}

fn solve<R, C, F>(
    client: &C,
    dtype: DType,
    domain: [&[f64]; 3],
    options: &CertifiedBisectionOptions,
    f: F,
) -> Result<Run>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    let [lower, upper, kappa] =
        domain.map(|values| fixture::tensor::<R>(values, &[values.len()], dtype, client.device()));
    let queries = RefCell::new(Vec::new());
    let mut bits = Vec::new();
    let point_bits = RefCell::new(Vec::new());
    let result = client.bisect_certified(
        |query| {
            queries.borrow_mut().push((
                query.owner.try_to_vec::<i64>()?,
                fixture::read(&query.point)?,
            ));
            point_bits
                .borrow_mut()
                .extend(fixture::bits(client, &query.point)?);
            f(query)
        },
        &lower?,
        &upper?,
        &kappa?,
        options,
    )?;
    bits.extend(fixture::bits(client, result.root_lower())?);
    bits.extend(fixture::bits(client, result.root_upper())?);
    bits.extend(point_bits.into_inner());
    let queries = queries.into_inner();
    let evaluated: usize = queries.iter().map(|(owner, _)| owner.len()).sum();
    assert_eq!(result.evaluated_points(), evaluated, "point accounting");
    assert!(result.evaluated_points() <= options.max_points);
    assert!(result.iterations() <= options.max_iterations);
    Ok(Run {
        status: result.status().try_to_vec::<i64>()?,
        owner: result.root_owner().try_to_vec::<i64>()?,
        lower: fixture::read(result.root_lower())?,
        upper: fixture::read(result.root_upper())?,
        iterations: result.iterations(),
        evaluated,
        queries,
        bits,
    })
}

fn solve_shapes<R, C>(
    client: &C,
    dtype: DType,
    rows: &[Row],
    options: &CertifiedBisectionOptions,
) -> Result<Run>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let params = shapes::params::<R>(rows, dtype, client.device())?;
    let lower: Vec<_> = rows.iter().map(|row| row.lower).collect();
    let upper: Vec<_> = rows.iter().map(|row| row.upper).collect();
    let kappa: Vec<_> = rows.iter().map(|row| row.shape.kappa()).collect();
    solve(client, dtype, [&lower, &upper, &kappa], options, |query| {
        shapes::evaluate(client, &params, &query.owner, &query.point)
    })
}

/// Every enclosure holds its row's root, inside the original domain.
fn contains(run: &Run, rows: &[Row], dtype: DType) {
    for (index, &owner) in run.owner.iter().enumerate() {
        let row = &rows[owner as usize];
        let (root_lower, root_upper) = row.root(dtype);
        let (lo, hi) = (run.lower[index], run.upper[index]);
        assert!(lo <= hi, "{row:?} inverted");
        assert!(
            lo <= root_upper && hi >= root_lower,
            "{row:?} excludes its root: [{lo}, {hi}]"
        );
        assert!(shapes::cast(row.lower, dtype) <= lo && hi <= shapes::cast(row.upper, dtype));
    }
}

fn known_rows() -> Vec<Row> {
    let row = |shape, parameter, lower, upper| Row {
        shape,
        parameter,
        lower,
        upper,
    };
    vec![
        row(Shape::Affine, 0.3, 0.0, 1.0),
        row(Shape::Affine, 0.71, 0.25, 0.9),
        row(Shape::Sqrt, 0.3, 0.0, 1.0),
        row(Shape::Sqrt, 0.77, 0.1, 1.0),
        row(Shape::Cubic, 0.123, -0.5, 2.0),
        row(Shape::Kinked, 0.3, 0.05, 1.0),
        row(Shape::Kinked, 0.55, 0.3, 0.9),
        // No root in the domain.
        row(Shape::Affine, 0.3, 0.5, 1.0),
        // A degenerate domain whose enclosure straddles zero.
        row(Shape::Affine, 0.3, 0.3, 0.3),
    ]
}

fn atol(dtype: DType) -> f64 {
    if dtype == DType::F32 { 1e-3 } else { 1e-10 }
}

fn known_roots<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let rows = known_rows();
    let slack = solve_shapes(client, dtype, &rows, &options(atol(dtype), SLACK))?;
    let midpoint = solve_shapes(client, dtype, &rows, &options(atol(dtype), MIDPOINT))?;
    for run in [&slack, &midpoint] {
        assert_eq!(run.status, [0, 0, 0, 0, 0, 0, 0, 1, 2], "{dtype:?}");
        assert_eq!(run.owner, [0, 1, 2, 3, 4, 5, 6]);
        contains(run, &rows, dtype);
        for (lo, hi) in run.lower.iter().zip(&run.upper) {
            assert!(hi - lo <= atol(dtype));
        }
    }
    // Rows freeze at different rounds, and each frozen row takes no further point.
    let rounds: Vec<_> = (0..7).map(|owner| slack.rounds(owner)).collect();
    assert!(rounds.iter().any(|&count| count != rounds[0]), "{rounds:?}");
    assert_eq!(slack.iterations, *rounds.iter().max().unwrap_or(&0));
    // One row can lose to a lucky midpoint run, but never to bisection's schedule.
    for (owner, row) in rows[..7].iter().enumerate() {
        let width = shapes::cast(row.upper, dtype) - shapes::cast(row.lower, dtype);
        let bound = (width / atol(dtype)).log2().ceil() as usize;
        assert!(
            slack.rounds(owner as i64) <= bound,
            "{dtype:?} owner {owner}"
        );
    }
    assert!(
        slack.evaluated < midpoint.evaluated,
        "{dtype:?}: {} slack points against {} midpoint points, rounds {rounds:?} against {:?}",
        slack.evaluated,
        midpoint.evaluated,
        (0..7)
            .map(|owner| midpoint.rounds(owner))
            .collect::<Vec<_>>()
    );
    Ok(())
}

/// The frozen midpoint fixtures keep their outcomes under secant probes.
fn fixture_cases<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    for case in fixture::cases() {
        let domain = [case.lower, case.upper, case.kappa].map(|raw| decode(raw, case.dtype));
        let options = CertifiedBisectionOptions {
            probe: SLACK,
            ..case.options()
        };
        let run = solve(
            client,
            case.dtype,
            [&domain[..1], &domain[1..2], &domain[2..]],
            &options,
            |query| callback::evaluate(client, case, query),
        )?;
        assert_eq!(run.status, [case.status], "{} status", case.id);
        assert_eq!(run.owner.len(), usize::from(case.certified), "{}", case.id);
        if case.certified {
            let (lo, hi) = (run.lower[0], run.upper[0]);
            assert!(
                domain[0] <= lo && lo <= hi && hi <= domain[1],
                "{}",
                case.id
            );
            assert!(lo <= decode(case.root_lower, case.dtype), "{}", case.id);
            assert!(hi >= decode(case.root_upper, case.dtype), "{}", case.id);
        }
        for (_, points) in &run.queries {
            for &x in points {
                assert!(
                    domain[0] <= x && x <= domain[1],
                    "{} probe outside",
                    case.id
                );
            }
        }
    }
    Ok(())
}

/// A wide enclosure band around the root forces uncertain signs at interpolated probes.
fn uncertain_band<R, C>(client: &C, dtype: DType) -> Result<Run>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let roots = [0.3, 0.6180339887, 0.45];
    let band = if dtype == DType::F32 { 1e-3 } else { 1e-7 };
    let root = fixture::tensor::<R>(&roots, &[3], dtype, client.device())?;
    let run = solve(
        client,
        dtype,
        [&[0.0, 0.1, 0.2], &[1.0, 0.9, 0.95], &[0.25, 0.5, 1.0]],
        &options(0.0, SLACK),
        |query| {
            let r = client.index_select(&root, 0, &query.owner)?;
            let value = client.sub(&query.point, &r)?;
            let near = client.lt(
                &client.abs(&value)?,
                &fixture::constant(client, &value, band)?,
            )?;
            let one = fixture::constant(client, &value, 1.0)?;
            Ok(CertifiedRootValues {
                lower: client.where_cond(&near, &client.neg(&one)?, &client.next_down(&value)?)?,
                upper: client.where_cond(&near, &one, &client.next_up(&value)?)?,
            })
        },
    )?;
    assert_eq!(run.status, [3, 3, 3], "{dtype:?}");
    for (index, &r) in roots.iter().enumerate() {
        let r = shapes::cast(r, dtype);
        assert!(run.lower[index] < r && r < run.upper[index]);
    }
    Ok(run)
}

/// An exact zero at a secant probe collapses the bracket onto the root.
fn exact_zero<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let root = shapes::cast(0.3, dtype);
    let run = solve(
        client,
        dtype,
        [&[0.0], &[1.0], &[0.5]],
        &options(0.0, SLACK),
        |query| callback::affine(client, &query.point, root),
    )?;
    assert_eq!(run.status, [0], "{dtype:?}");
    assert_eq!((run.lower[0], run.upper[0]), (root, root));
    Ok(())
}

/// Iteration and point limits count executed rounds and points exactly.
fn limits<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let rows = &known_rows()[..7];
    let full = solve_shapes(client, dtype, rows, &options(atol(dtype), SLACK))?;
    for max_iterations in 0..=full.iterations + 1 {
        let limited = CertifiedBisectionOptions {
            max_iterations,
            ..options(atol(dtype), SLACK)
        };
        let run = solve_shapes(client, dtype, rows, &limited)?;
        assert_eq!(run.iterations, max_iterations.min(full.iterations));
        assert_eq!(run.queries.len(), 2 + run.iterations);
        let limited = max_iterations < full.iterations;
        assert_eq!(run.status.contains(&4), limited, "{max_iterations}");
        contains(&run, rows, dtype);
        // A prefix of the full run executes the same probes.
        assert_eq!(run.queries[..], full.queries[..run.queries.len()]);
    }
    for max_points in 2 * rows.len()..full.evaluated + 2 {
        let limited = CertifiedBisectionOptions {
            max_points,
            ..options(atol(dtype), SLACK)
        };
        let run = solve_shapes(client, dtype, rows, &limited)?;
        assert!(run.evaluated <= max_points);
        assert!(run.status.iter().all(|&code| code == 0 || code == 4));
        contains(&run, rows, dtype);
    }
    Ok(())
}

/// Second points never cost a probe round that midpoint probes could afford.
fn point_budget_rounds<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let rows = &known_rows()[..7];
    let count = rows.len();
    let (mut paired, mut compared) = (false, 0);
    for max_iterations in [1, 2, 400] {
        for max_points in 2 * count..2 * count + count * 8 {
            let limited = |probe| CertifiedBisectionOptions {
                max_iterations,
                max_points,
                ..options(0.0, probe)
            };
            let slack = solve_shapes(client, dtype, rows, &limited(SLACK))?;
            let midpoint = solve_shapes(client, dtype, rows, &limited(MIDPOINT))?;
            assert!(slack.evaluated <= max_points);
            contains(&slack, rows, dtype);
            paired |= slack.evaluated > 2 * count + count * slack.iterations;
            // The reserve pays midpoint rounds for every row still pending. A midpoint run
            // whose rows all stayed pending therefore never runs more rounds.
            if midpoint.evaluated == 2 * count + count * midpoint.iterations {
                compared += 1;
                assert!(
                    slack.iterations >= midpoint.iterations,
                    "{dtype:?} max_iterations {max_iterations} max_points {max_points}: {} slack rounds against {} midpoint rounds",
                    slack.iterations,
                    midpoint.iterations
                );
            }
        }
    }
    assert!(paired, "{dtype:?}: no budget admitted a second point");
    assert!(
        compared > count * 4,
        "{dtype:?}: {compared} comparable budgets"
    );
    // One round per call, as callers that split a session budget across calls run it.
    let first = |probe| CertifiedBisectionOptions {
        max_iterations: 1,
        max_points: 2 * rows.len() + rows.len(),
        ..options(0.0, probe)
    };
    let slack = solve_shapes(client, dtype, rows, &first(SLACK))?;
    assert_eq!(slack.iterations, 1);
    assert_eq!(slack.evaluated, 3 * rows.len());
    Ok(())
}

/// Exact roots on simple floats reach an exact zero after secant steps leave the grid.
fn dyadic_roots<R, C>(client: &C, dtype: DType) -> Result<Run>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let roots = [0.125, 0.375, -0.25, 0.5];
    // The radius covers the rounding of the cubic. The band of uncertain signs is wider.
    let (band, radius) = if dtype == DType::F32 {
        (1e-5, 1e-6)
    } else {
        (1e-12, 1e-15)
    };
    let root = fixture::tensor::<R>(&roots, &[4], dtype, client.device())?;
    // The cubic keeps secant points off the root. Off-grid domains keep midpoints off it.
    let run = solve(
        client,
        dtype,
        [
            &[0.01, 0.3, -0.61, 0.07],
            &[0.9, 0.47, 0.2, 0.93],
            &[0.5, 1.0, 0.25, 0.75],
        ],
        &options(0.0, SLACK),
        |query| {
            let r = client.index_select(&root, 0, &query.owner)?;
            // Sterbenz makes the difference exact near the root, so only the root gives zero.
            let t = client.sub(&query.point, &r)?;
            let value = client.add(&client.mul(&client.mul(&t, &t)?, &t)?, &t)?;
            let zero = fixture::constant(client, &t, 0.0)?;
            let one = fixture::constant(client, &t, 1.0)?;
            let near = client.lt(&client.abs(&t)?, &fixture::constant(client, &t, band)?)?;
            let exact = client.eq(&t, &zero)?;
            let radius = fixture::constant(client, &t, radius)?;
            let lower = client.where_cond(
                &near,
                &client.neg(&one)?,
                &client.next_down(&client.sub(&value, &radius)?)?,
            )?;
            let upper =
                client.where_cond(&near, &one, &client.next_up(&client.add(&value, &radius)?)?)?;
            Ok(CertifiedRootValues {
                lower: client.where_cond(&exact, &zero, &lower)?,
                upper: client.where_cond(&exact, &zero, &upper)?,
            })
        },
    )?;
    assert_eq!(run.status, [0, 0, 0, 0], "{dtype:?} {run:?}");
    for (index, &r) in roots.iter().enumerate() {
        assert_eq!(
            (run.lower[index], run.upper[index]),
            (r, r),
            "{dtype:?} row {index}"
        );
    }
    Ok(run)
}

/// Random increasing residuals never need more rounds than bisection's schedule.
fn rounds_bound<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let mut random = Lcg(0x5eed_0000 + dtype.size_in_bytes() as u64);
    let (low, high): (f64, f64) = if dtype == DType::F32 {
        (1e-3, 1e-2)
    } else {
        (1e-11, 1e-8)
    };
    for _ in 0..6 {
        let rows: Vec<_> = (0..24).map(|_| random.row(dtype)).collect();
        // The tolerance stays clear of exact power-of-two ratios to every width.
        let w = loop {
            let w = shapes::cast(low * (high / low).powf(random.next()), dtype);
            let clear = rows.iter().all(|row| {
                let fraction = ((row.upper - row.lower) / w).log2().fract();
                fraction > 1e-6 && fraction < 1.0 - 1e-6
            });
            if clear {
                break w;
            }
        };
        let run = solve_shapes(client, dtype, &rows, &options(w, SLACK))?;
        assert!(run.status.iter().all(|&status| status == 0), "{run:?}");
        contains(&run, &rows, dtype);
        let mut bounds = HashMap::new();
        for (owner, row) in rows.iter().enumerate() {
            let bound = ((row.upper - row.lower) / w).log2().ceil() as usize;
            let rounds = run.rounds(owner as i64);
            assert!(rounds <= bound, "{row:?}: {rounds} rounds against {bound}");
            bounds.insert(owner, bound);
        }
        assert!(run.iterations <= bounds.values().copied().max().unwrap_or(0));
    }
    Ok(())
}

/// Bits of every result and probe, in a fixed order across all slack cases.
#[cfg(feature = "cuda")]
pub fn trace<R, C>(client: &C) -> Result<Vec<Run>>
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let mut runs = Vec::new();
    for dtype in [DType::F32, DType::F64] {
        let rows = known_rows();
        runs.push(solve_shapes(
            client,
            dtype,
            &rows,
            &options(atol(dtype), SLACK),
        )?);
        runs.push(uncertain_band(client, dtype)?);
        runs.push(dyadic_roots(client, dtype)?);
        let mut random = Lcg(0x0b17 + dtype.size_in_bytes() as u64);
        let rows: Vec<_> = (0..64).map(|_| random.row(dtype)).collect();
        runs.push(solve_shapes(
            client,
            dtype,
            &rows,
            &options(atol(dtype), SLACK),
        )?);
    }
    Ok(runs)
}

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    fixture_cases(client)?;
    for dtype in [DType::F32, DType::F64] {
        known_roots(client, dtype)?;
        uncertain_band(client, dtype)?;
        exact_zero(client, dtype)?;
        limits(client, dtype)?;
        point_budget_rounds(client, dtype)?;
        dyadic_roots(client, dtype)?;
        rounds_bound(client, dtype)?;
        let rows = known_rows();
        // Repeated runs match bit for bit, including every probe point.
        let first = solve_shapes(client, dtype, &rows, &options(atol(dtype), SLACK))?;
        let second = solve_shapes(client, dtype, &rows, &options(atol(dtype), SLACK))?;
        assert_eq!(first, second);
    }
    Ok(())
}
