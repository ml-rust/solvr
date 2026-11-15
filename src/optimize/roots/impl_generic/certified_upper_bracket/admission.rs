use super::super::certified::{metadata, native};
use crate::DType;
use crate::optimize::roots::traits::CertifiedUpperBracketOptions;
use crate::optimize::{OptimizeError, OptimizeResult};
use native::RootClient;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn metadata<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    limit: &Tensor<R>,
    options: &CertifiedUpperBracketOptions,
) -> OptimizeResult<usize>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if !matches!(lower.dtype(), DType::F32 | DType::F64) {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Certified upper bracket dtype {:?} requires F32 or F64. Select a supported dtype.",
                lower.dtype()
            ),
        });
    }
    for (name, value) in [
        ("lower", lower),
        ("initial_upper", upper),
        ("upper_limit", limit),
    ] {
        metadata::tensor(client, value, lower.numel(), lower.dtype(), name)?;
    }
    metadata::count(lower.numel(), lower.dtype())?;
    metadata::count(lower.numel(), DType::I64)?;
    let initial = lower
        .numel()
        .checked_mul(2)
        .ok_or_else(|| OptimizeError::InvalidInput {
            context:
                "Certified upper bracket initial query count overflows usize. Split the batch."
                    .into(),
        })?;
    metadata::count(initial, lower.dtype())?;
    metadata::count(initial, DType::I64)?;
    if initial > options.max_points {
        return Err(OptimizeError::InvalidParameter {
            parameter: "max_points".into(),
            message: format!(
                "Certified upper bracket initial admission needs {initial} points, but max_points={}. Increase the budget or split the batch.",
                options.max_points
            ),
        });
    }
    if i64::try_from(options.max_expansions).is_err() {
        return Err(OptimizeError::InvalidParameter {
            parameter: "max_expansions".into(),
            message: format!(
                "Certified upper bracket max_expansions={} exceeds I64 counting capacity. Reduce the expansion budget.",
                options.max_expansions
            ),
        });
    }
    Ok(initial)
}

pub(super) fn values<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    limit: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let invalid = client.maximum(
        &native::nonfinite(client, lower)?,
        &native::nonfinite(client, upper)?,
    )?;
    let invalid = client.maximum(&invalid, &native::nonfinite(client, limit)?)?;
    let invalid = client.maximum(&invalid, &client.gt(lower, upper)?)?;
    let invalid = client.maximum(&invalid, &client.gt(upper, limit)?)?;
    if native::any(client, &invalid)? {
        return Err(OptimizeError::InvalidInput {
            context: "Certified upper bracket inputs require finite lower <= initial_upper <= upper_limit. Correct the input bounds.".into(),
        });
    }
    Ok(())
}
