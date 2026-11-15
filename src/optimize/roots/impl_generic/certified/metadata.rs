use crate::DType;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn tensor<R, C>(
    client: &C,
    value: &Tensor<R>,
    len: usize,
    dtype: DType,
    name: &str,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if value.ndim() != 1
        || value.numel() != len
        || value.dtype() != dtype
        || !client.device().is_same(value.device())
    {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Certified root {name} has shape {:?}, dtype {:?}, or a foreign device. Supply [{len}] {dtype:?} on the configured client.",
                value.shape(),
                value.dtype(),
            ),
        });
    }
    Ok(())
}

pub(crate) fn count(count: usize, dtype: DType) -> OptimizeResult<()> {
    let bytes = count.checked_mul(dtype.size_in_bytes());
    if u32::try_from(count).is_err() || bytes.is_none_or(|n| n > isize::MAX as usize) {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Certified root count {count} exceeds native indexing or allocation limits for {dtype:?}. Split the batch."
            ),
        });
    }
    Ok(())
}
