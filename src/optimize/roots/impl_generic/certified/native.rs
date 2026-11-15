//! Native storage, masks, and checked-arithmetic adaptation.

use crate::DType;
use crate::common::arithmetic::ArithmeticError;
pub(crate) use crate::optimize::roots::traits::certified_client::RootClient;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn arithmetic(error: ArithmeticError) -> OptimizeError {
    OptimizeError::NumericalError {
        message: error.to_string(),
    }
}

pub(crate) fn owned<R, C>(client: &C, value: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let output = Tensor::empty(value.shape(), value.dtype(), client.device())?;
    client.copy_into(&output, value)?;
    Ok(output)
}

pub(crate) fn replace<R, C>(
    client: &C,
    original: &Tensor<R>,
    indices: &Tensor<R>,
    values: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // Scatter destinations own storage even when a backend mutates its input.
    let destination = owned(client, original)?;
    Ok(client.scatter(&destination, 0, indices, values)?)
}

pub(crate) fn indices<R, C>(client: &C, mask: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let indices = client.nonzero(mask)?;
    let flat = indices.reshape(&[indices.numel()])?;
    Ok(client.sort(&flat, 0, false)?)
}

pub(crate) fn any<R, C>(client: &C, mask: &Tensor<R>) -> OptimizeResult<bool>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    if mask.numel() == 0 {
        return Ok(false);
    }
    let reduced = client.any(mask, &[], false)?;
    match mask.dtype() {
        DType::F32 => Ok(reduced.item::<f32>()? != 0.0),
        DType::F64 => Ok(reduced.item::<f64>()? != 0.0),
        _ => Err(OptimizeError::NumericalError {
            message:
                "Certified root control mask requires F32 or F64. Check the native mask dtype."
                    .into(),
        }),
    }
}

pub(crate) fn nonfinite<R, C>(client: &C, value: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    Ok(client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?)
}
