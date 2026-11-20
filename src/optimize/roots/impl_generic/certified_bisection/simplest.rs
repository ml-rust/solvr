//! The float with the fewest significant bits inside a certified root bracket.
//!
//! An exact root can only be certified by evaluating that exact float. Simple floats such
//! as dyadic rationals are the likeliest exact roots, so the noise-band endgame probes them.

use super::native::{self, RootClient};
use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The simplest float strictly inside each bracket, and a mask of usable lanes.
///
/// A lane is usable when that float is finite, strictly inside, and unequal to the midpoint.
/// `width` is the bracket width rounded up.
pub(super) fn simplest<R, C>(
    client: &C,
    bracket: [&Tensor<R>; 2],
    width: &Tensor<R>,
    midpoint: &Tensor<R>,
) -> OptimizeResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let [lower, upper] = bracket;
    let digits = if lower.dtype() == DType::F32 { 23 } else { 52 };
    // The spacing above a normal width, scaled by its significand bits, is the largest
    // power of two at most that width. Both steps are exact.
    let spacing = client.sub(&client.next_up(width)?, width)?;
    let unit = client.mul_scalar(&spacing, 2f64.powi(digits))?;
    // Division and multiplication by a power of two are exact, so each candidate is
    // the smallest multiple of its step above `lower`.
    let above = |step: &Tensor<R>| -> OptimizeResult<Tensor<R>> {
        let count = client.add_scalar(&client.floor(&client.div(lower, step)?)?, 1.0)?;
        Ok(client.mul(&count, step)?)
    };
    let coarse = above(&client.mul_scalar(&unit, 2.0)?)?;
    let fine = above(&unit)?;
    // The bracket is narrower than twice the unit, so it holds at most one multiple of
    // the coarse step. Without one, it holds at most one multiple of the unit.
    let point = client.where_cond(&client.lt(&coarse, upper)?, &coarse, &fine)?;
    let zero = Tensor::zeros(point.shape(), point.dtype(), client.device())?;
    let finite = client.eq(&native::nonfinite(client, &point)?, &zero)?;
    let usable = client.mul(
        &client.mul(&client.gt(&point, lower)?, &client.lt(&point, upper)?)?,
        &client.mul(&client.ne(&point, midpoint)?, &finite)?,
    )?;
    Ok((point, usable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::prelude::*;

    type T = Tensor<CpuRuntime>;
    type Result<V = ()> = std::result::Result<V, Box<dyn std::error::Error>>;

    fn tensor(client: &CpuClient, values: &[f64], dtype: DType) -> Result<T> {
        let shape = [values.len()];
        Ok(match dtype {
            DType::F32 => T::from_slice(
                &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &shape,
                client.device(),
            )?,
            _ => T::from_slice(values, &shape, client.device())?,
        })
    }

    fn read(value: &T) -> Result<Vec<f64>> {
        Ok(match value.dtype() {
            DType::F32 => value
                .try_to_vec::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect(),
            _ => value.try_to_vec::<f64>()?,
        })
    }

    #[test]
    fn simplest_takes_the_float_with_fewest_significant_bits() -> Result {
        let client = CpuClient::new(CpuDevice::new());
        for dtype in [DType::F32, DType::F64] {
            let ulp = if dtype == DType::F32 {
                f64::from(f32::EPSILON)
            } else {
                f64::EPSILON
            };
            // An off-grid bracket around 0.125, one in a single binade, one across zero,
            // one whose simplest float is its midpoint, and one without an interior multiple.
            let lower = [0.125 - 3.0 * ulp / 16.0, 0.3, -0.1, 0.25, 0.0];
            let upper = [0.125 + 3.0 * ulp / 8.0, 0.9, 0.3, 0.75, 1.0];
            let lower = tensor(&client, &lower, dtype)?;
            let upper = tensor(&client, &upper, dtype)?;
            let width = client.next_up(&client.sub(&upper, &lower)?)?;
            let midpoint = client.midpoint(&lower, &upper)?;
            let (point, usable) = simplest(&client, [&lower, &upper], &width, &midpoint)?;
            assert_eq!(&read(&point)?[..4], [0.125, 0.5, 0.0, 0.5], "{dtype:?}");
            assert_eq!(read(&usable)?, [1.0, 1.0, 1.0, 0.0, 0.0], "{dtype:?}");
        }
        Ok(())
    }
}
