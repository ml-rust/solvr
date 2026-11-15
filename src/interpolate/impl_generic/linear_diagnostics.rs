//! Closed-support Linear diagnostics with same-device arithmetic admission.

use crate::DType;
use crate::common::arithmetic::envelope;
use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::linear_diagnostics::LinearInterp1dReceipt;
use numr::ops::{CompareOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{FloatArithmeticContract, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::interp1d::{linear, validation};

mod error;
mod receipts;

/// Evaluate ordinary Linear values with checked independent absolute-error receipts.
///
/// Requires a declared IEEE basic-operation provider contract before testing the current arithmetic mode.
/// Backend-defined provider arithmetic rejects before floating input checks or sampled mode probes.
pub fn interp1d_linear_with_receipts_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<LinearInterp1dReceipt<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + TwoSumOps<R> + RuntimeClient<R>,
{
    validation::check_metadata(client, x, y, query)?;
    if R::float_arithmetic_contract(x.dtype()) != FloatArithmeticContract::Ieee754Basic {
        return Err(InterpolateError::UnsupportedArithmetic {
            dtype: x.dtype(),
            failed_probes: vec!["provider_contract"],
        });
    }
    envelope::check(client, x.dtype())?;
    validation::check_values(client, x, y, query)?;
    if query.numel() == 0 {
        let values = Tensor::empty(query.shape(), query.dtype(), client.device())?;
        let indices = client.searchsorted(x, query, false)?;
        return Ok(LinearInterp1dReceipt {
            absolute_error: values.clone(),
            at_lower: values.clone(),
            at_upper: values.clone(),
            constant: values.clone(),
            values,
            lower_indices: indices.clone(),
            upper_indices: indices,
        });
    }
    check_closed_support(client, x, query)?;
    let evaluation = linear::evaluate_captured(client, x, y, query)?;
    receipts::assemble(client, query, evaluation)
}

fn control_value<R: Runtime<DType = DType>>(value: &Tensor<R>) -> InterpolateResult<f64> {
    match value.dtype() {
        DType::F32 => Ok(f64::from(value.item::<f32>()?)),
        DType::F64 => Ok(value.item::<f64>()?),
        _ => Err(InterpolateError::NumericalError {
            message: "Linear receipt control value requires F32 or F64".into(),
        }),
    }
}

fn check_closed_support<R, C>(client: &C, x: &Tensor<R>, query: &Tensor<R>) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let first = x.narrow(0, 0, 1)?;
    let last = x.narrow(0, x.numel() - 1, 1)?;
    let outside = client.maximum(&client.lt(query, &first)?, &client.gt(query, &last)?)?;
    if validation::any(client, &outside)? {
        let absent =
            Tensor::full_scalar(query.shape(), query.dtype(), f64::INFINITY, client.device())?;
        let invalid_queries = client.where_cond(&outside, query, &absent)?;
        let point = client.min(&invalid_queries, &[], false)?;
        return Err(InterpolateError::OutOfDomain {
            point: control_value(&point)?,
            min: control_value(&first)?,
            max: control_value(&last)?,
            context: "Linear receipts require queries within the closed original grid support"
                .into(),
        });
    }
    Ok(())
}
