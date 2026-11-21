//! Interval acos and log against 100-digit references.

use super::elementary_references::{ACOS, LOG};
use super::support::{Client, Result, failure, range};
use numr::ops::CorrectlyRoundedOps;
use numr::prelude::*;
use solvr::common::arithmetic::{Interval, with_interval_arithmetic};

const PI_LOWER: u64 = 0x400921fb54442d18;
const PI_UPPER: u64 = 0x400921fb54442d19;

/// Clients that also provide numr's correctly rounded kernels.
pub trait ElementaryClient<R: Runtime<DType = DType>>: Client<R> + CorrectlyRoundedOps<R> {}
impl<R, C> ElementaryClient<R> for C
where
    R: Runtime<DType = DType>,
    C: Client<R> + CorrectlyRoundedOps<R>,
{
}

fn inputs<R: Runtime<DType = DType>>(
    rows: &[(u64, u64, u64)],
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    let values: Vec<f64> = rows.iter().map(|row| f64::from_bits(row.0)).collect();
    Tensor::from_slice(&values, &[values.len()], device)
}

fn endpoints<R: Runtime<DType = DType>>(value: &Interval<R>) -> Result<(Vec<f64>, Vec<f64>)> {
    Ok((
        value.lower().try_to_vec::<f64>()?,
        value.upper().try_to_vec::<f64>()?,
    ))
}

/// Check containment of the true bracket and at most one extra step on each side.
fn tight(lower: f64, upper: f64, down: f64, up: f64, lane: &str) {
    assert!(
        lower <= down && upper >= up,
        "{lane}: [{lower:e}, {upper:e}] excludes [{down:e}, {up:e}]"
    );
    assert!(
        lower >= down.next_down() && upper <= up.next_up(),
        "{lane}: [{lower:e}, {upper:e}] is wider than one step past [{down:e}, {up:e}]"
    );
}

/// Point and range intervals enclose the 100-digit acos references.
pub fn acos<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: ElementaryClient<R>,
{
    let device = client.device();
    let points = inputs::<R>(&ACOS, device)?;
    let result =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "acos points", |context| {
            let input = context.point(&points)?;
            context.acos(&input)
        })?;
    let (lower, upper) = endpoints(&result)?;
    for (index, &(input, down, up)) in ACOS.iter().enumerate() {
        let lane = format!("acos lane {index} input {:e}", f64::from_bits(input));
        tight(
            lower[index],
            upper[index],
            f64::from_bits(down),
            f64::from_bits(up),
            &lane,
        );
    }
    // Exact points: acos(-1) is the F64 pi pair, acos(1) is [0, 0].
    assert_eq!(lower[0].to_bits(), PI_LOWER);
    assert_eq!(upper[0].to_bits(), PI_UPPER);
    assert_eq!(lower[1].to_bits(), 0);
    assert_eq!(upper[1].to_bits(), 0);

    // Ranges swap sides: [lo, hi] maps to [acos(hi), acos(lo)].
    let pairs = [(5, 4), (0, 1), (7, 6), (13, 12), (2, 4)];
    let range_lower: Vec<f64> = pairs.iter().map(|p| f64::from_bits(ACOS[p.0].0)).collect();
    let range_upper: Vec<f64> = pairs.iter().map(|p| f64::from_bits(ACOS[p.1].0)).collect();
    let range_lower = Tensor::<R>::from_slice(&range_lower, &[pairs.len()], device)?;
    let range_upper = Tensor::<R>::from_slice(&range_upper, &[pairs.len()], device)?;
    let ranges =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "acos ranges", |context| {
            let input = context.bounds(&range_lower, &range_upper)?;
            context.acos(&input)
        })?;
    let (lower, upper) = endpoints(&ranges)?;
    for (index, &(low, high)) in pairs.iter().enumerate() {
        let lane = format!("acos range {index}");
        tight(
            lower[index],
            upper[index],
            f64::from_bits(ACOS[high].1),
            f64::from_bits(ACOS[low].2),
            &lane,
        );
    }
    Ok(())
}

/// Point and range intervals enclose the 100-digit log references.
pub fn log<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: ElementaryClient<R>,
{
    let device = client.device();
    let points = inputs::<R>(&LOG, device)?;
    let result =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "log points", |context| {
            let input = context.point(&points)?;
            context.log_positive(&input)
        })?;
    let (lower, upper) = endpoints(&result)?;
    for (index, &(input, down, up)) in LOG.iter().enumerate() {
        let lane = format!("log lane {index} input {:e}", f64::from_bits(input));
        tight(
            lower[index],
            upper[index],
            f64::from_bits(down),
            f64::from_bits(up),
            &lane,
        );
    }
    // log(1) = 0 spans one subnormal step on each side.
    assert_eq!(lower[0].to_bits(), (-f64::from_bits(1)).to_bits());
    assert_eq!(upper[0].to_bits(), 1);

    // Ranges keep sides: [lo, hi] maps to [log(lo), log(hi)].
    let pairs = [(2, 3), (7, 10), (6, 5)];
    let range_lower: Vec<f64> = pairs.iter().map(|p| f64::from_bits(LOG[p.0].0)).collect();
    let range_upper: Vec<f64> = pairs.iter().map(|p| f64::from_bits(LOG[p.1].0)).collect();
    let range_lower = Tensor::<R>::from_slice(&range_lower, &[pairs.len()], device)?;
    let range_upper = Tensor::<R>::from_slice(&range_upper, &[pairs.len()], device)?;
    let ranges =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "log ranges", |context| {
            let input = context.bounds(&range_lower, &range_upper)?;
            context.log_positive(&input)
        })?;
    let (lower, upper) = endpoints(&ranges)?;
    for (index, &(low, high)) in pairs.iter().enumerate() {
        let lane = format!("log range {index}");
        tight(
            lower[index],
            upper[index],
            f64::from_bits(LOG[low].1),
            f64::from_bits(LOG[high].2),
            &lane,
        );
    }
    Ok(())
}

/// Endpoints outside the domain reject at completion. Empty inputs complete.
pub fn domain<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: ElementaryClient<R>,
{
    let device = client.device();
    let below = (-1.0f64).next_down();
    let above = 1.0f64.next_up();
    let acos_cases = [(below, 0.0), (0.0, above), (2.0, 3.0), (-3.0, -2.0)];
    for (low, high) in acos_cases {
        let low = Tensor::<R>::from_slice(&[0.25, low], &[2], device)?;
        let high = Tensor::<R>::from_slice(&[0.5, high], &[2], device)?;
        range(
            failure(with_interval_arithmetic::<R, _, _, _>(
                client,
                DType::F64,
                "acos domain",
                |context| {
                    let input = context.bounds(&low, &high)?;
                    context.acos(&input)
                },
            )),
            "acos domain",
        );
    }
    for (low, high) in [(0.0, 1.0), (-0.0, 1.0), (-2.0, -1.0), (-1.0, 2.0)] {
        let low = Tensor::<R>::from_slice(&[1.0, low], &[2], device)?;
        let high = Tensor::<R>::from_slice(&[2.0, high], &[2], device)?;
        range(
            failure(with_interval_arithmetic::<R, _, _, _>(
                client,
                DType::F64,
                "log domain",
                |context| {
                    let input = context.bounds(&low, &high)?;
                    context.log_positive(&input)
                },
            )),
            "log domain",
        );
    }

    let empty = Tensor::<R>::empty(&[0, 3], DType::F64, device)?;
    let (angle, logarithm) =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "empty", |context| {
            let input = context.point(&empty)?;
            Ok((context.acos(&input)?, context.log_positive(&input)?))
        })?;
    for value in [&angle, &logarithm] {
        assert_eq!(value.lower().shape(), &[0, 3]);
        assert_eq!(value.upper().shape(), &[0, 3]);
    }
    Ok(())
}

/// Endpoint bits of acos and log over every reference input, for backend parity.
pub fn bits<R, C>(client: &C) -> Result<Vec<u64>>
where
    R: Runtime<DType = DType>,
    C: ElementaryClient<R>,
{
    let device = client.device();
    let angles = inputs::<R>(&ACOS, device)?;
    let logs = inputs::<R>(&LOG, device)?;
    let (angle, logarithm) =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, "parity", |context| {
            let angles = context.point(&angles)?;
            let logs = context.point(&logs)?;
            Ok((context.acos(&angles)?, context.log_positive(&logs)?))
        })?;
    let mut bits = Vec::new();
    for value in [&angle, &logarithm] {
        let (lower, upper) = endpoints(value)?;
        bits.extend(lower.into_iter().chain(upper).map(f64::to_bits));
    }
    Ok(bits)
}
