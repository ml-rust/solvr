//! Input metadata and allocation admission before native snapshots.

pub(crate) use super::super::certified::metadata::{count, tensor};
use super::native::{self, RootClient};
use crate::DType;
use crate::optimize::roots::traits::CertifiedBisectionOptions;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn metadata<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if !matches!(lower.dtype(), DType::F32 | DType::F64) {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Certified root dtype {:?} requires F32 or F64. Select a supported dtype.",
                lower.dtype()
            ),
        });
    }
    for (name, value) in [
        ("lower", lower),
        ("upper", upper),
        ("derivative_lower", derivative),
    ] {
        tensor(client, value, lower.numel(), lower.dtype(), name)?;
    }
    tolerances(options.atol, options.rtol)?;
    count(lower.numel(), lower.dtype())?;
    count(lower.numel(), DType::I64)?;
    let initial = lower
        .numel()
        .checked_mul(2)
        .ok_or_else(|| OptimizeError::InvalidInput {
            context: "Certified root initial query count overflows usize. Split the batch.".into(),
        })?;
    count(initial, lower.dtype())?;
    count(initial, DType::I64)?;
    if initial > options.max_points {
        return Err(OptimizeError::InvalidParameter {
            parameter: "max_points".into(),
            message: format!(
                "Certified root initial admission needs {initial} points, but max_points={}. Increase the budget or split the batch.",
                options.max_points
            ),
        });
    }
    Ok(())
}

pub(crate) fn tolerances(atol: f64, rtol: f64) -> OptimizeResult<()> {
    for (name, value) in [("atol", atol), ("rtol", rtol)] {
        if !value.is_finite() || value < 0.0 {
            return Err(OptimizeError::InvalidParameter {
                parameter: name.into(),
                message: format!(
                    "Certified root {name}={value} requires a finite nonnegative value. Supply a valid tolerance."
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn values<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let zero = Tensor::zeros(&[1], lower.dtype(), client.device())?;
    let finite = client.maximum(
        &native::nonfinite(client, lower)?,
        &native::nonfinite(client, upper)?,
    )?;
    let invalid = client.maximum(&finite, &native::nonfinite(client, derivative)?)?;
    let invalid = client.maximum(&invalid, &client.gt(lower, upper)?)?;
    let invalid = client.maximum(&invalid, &client.le(derivative, &zero)?)?;
    if native::any(client, &invalid)? {
        return Err(OptimizeError::InvalidInput {
            context: "Certified root inputs require finite lower <= upper and finite derivative_lower > 0. Correct the input bounds.".into(),
        });
    }
    Ok(())
}
