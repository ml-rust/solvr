//! Intersect finite brackets with outward-rounded physical residual corrections.

use crate::DType;
use crate::common::arithmetic::{ArithmeticError, ArithmeticResult, Interval, IntervalContext};
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) struct CorrectionInputs<'a, R: Runtime<DType = DType>> {
    pub lower: &'a Tensor<R>,
    pub upper: &'a Tensor<R>,
    pub point: &'a Tensor<R>,
    pub residual_lower: &'a Tensor<R>,
    pub residual_upper: &'a Tensor<R>,
    pub derivative_lower: &'a Tensor<R>,
    pub active: &'a Tensor<R>,
}

/// Intersect a bracket using physical residual bounds and a positive derivative floor.
///
/// The caller establishes continuity and the derivative floor throughout the old bracket.
/// Raw operations permit outward infinities until intersection with finite old endpoints.
pub(super) fn intersect<R, C>(
    ctx: &mut IntervalContext<'_, R, C>,
    inputs: CorrectionInputs<'_, R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let CorrectionInputs {
        lower,
        upper,
        point,
        residual_lower,
        residual_upper,
        derivative_lower,
        active,
    } = inputs;
    let client = ctx.client();
    if lower.ndim() != 1 || !lower.device().is_same(client.device()) {
        return Err(ArithmeticError::InvalidBounds);
    }
    for value in [
        upper,
        point,
        residual_lower,
        residual_upper,
        derivative_lower,
        active,
    ] {
        if value.shape() != lower.shape()
            || value.dtype() != lower.dtype()
            || !value.device().is_same(lower.device())
        {
            return Err(ArithmeticError::InvalidBounds);
        }
    }
    if !matches!(lower.dtype(), DType::F32 | DType::F64) {
        return Err(ArithmeticError::UnsupportedDType {
            dtype: lower.dtype(),
        });
    }
    // Capture admission also checks the context dtype and native allocation limits.
    let old = ctx.bounds(lower, upper)?;
    let lower = old.lower();
    let upper = old.upper();
    if lower.numel() == 0 {
        return Ok(old);
    }
    let zero = Tensor::zeros(lower.shape(), lower.dtype(), client.device())?;
    let one = Tensor::ones(lower.shape(), lower.dtype(), client.device())?;
    check_finite(client, lower)?;
    check_finite(client, upper)?;
    reject(client, &client.gt(lower, upper)?)?;
    let valid_active = client.maximum(&client.eq(active, &zero)?, &client.eq(active, &one)?)?;
    reject(client, &client.ne(&valid_active, &one)?)?;

    // Selection precedes every active-only check and eager arithmetic operation.
    let point = client.where_cond(active, point, lower)?;
    let a = client.where_cond(active, residual_lower, &zero)?;
    let b = client.where_cond(active, residual_upper, &zero)?;
    let kappa = client.where_cond(active, derivative_lower, &one)?;
    for value in [&point, &a, &b, &kappa] {
        check_finite(client, value)?;
    }
    for invalid in [
        client.lt(&point, lower)?,
        client.gt(&point, upper)?,
        client.gt(&a, &b)?,
        client.le(&kappa, &zero)?,
    ] {
        reject(client, &invalid)?;
    }
    let left_numerator = client.maximum(&b, &zero)?;
    let right_numerator = client.maximum(&client.neg(&a)?, &zero)?;
    let left_zero = client.eq(&left_numerator, &zero)?;
    let right_zero = client.eq(&right_numerator, &zero)?;
    // Zero sides use harmless finite operands throughout eager arithmetic.
    let left_q = client.div(
        &client.where_cond(&left_zero, &one, &left_numerator)?,
        &client.where_cond(&left_zero, &one, &kappa)?,
    )?;
    let right_q = client.div(
        &client.where_cond(&right_zero, &one, &right_numerator)?,
        &client.where_cond(&right_zero, &one, &kappa)?,
    )?;
    for quotient in [&left_q, &right_q] {
        check_nonnegative_extended(client, quotient, &zero)?;
    }
    // Nearest-even division followed by next_up encloses the positive exact quotient, including underflow and overflow.
    let left_q = client.next_up(&left_q)?;
    let right_q = client.next_up(&right_q)?;
    for quotient in [&left_q, &right_q] {
        check_nonnegative_extended(client, quotient, &zero)?;
    }
    let candidate_lower =
        client.next_down(&client.sub(&client.where_cond(&left_zero, &zero, &point)?, &left_q)?)?;
    let candidate_upper =
        client.next_up(&client.add(&client.where_cond(&right_zero, &zero, &point)?, &right_q)?)?;
    let candidate_lower = client.where_cond(&left_zero, &point, &candidate_lower)?;
    let candidate_upper = client.where_cond(&right_zero, &point, &candidate_upper)?;
    let infinity =
        Tensor::full_scalar(lower.shape(), lower.dtype(), f64::INFINITY, client.device())?;
    // Only negative infinity belongs to a lower bound. Only positive infinity belongs to an upper bound.
    check_not_nan(client, &candidate_lower)?;
    check_not_nan(client, &candidate_upper)?;
    reject(client, &client.eq(&candidate_lower, &infinity)?)?;
    reject(
        client,
        &client.eq(&candidate_upper, &client.neg(&infinity)?)?,
    )?;
    let clipped_lower = client.maximum(lower, &candidate_lower)?;
    let clipped_upper = client.minimum(upper, &candidate_upper)?;
    // Exact zero corrections preserve candidate bits, including signed zero.
    let clipped_lower = client.where_cond(&left_zero, &point, &clipped_lower)?;
    let clipped_upper = client.where_cond(&right_zero, &point, &clipped_upper)?;
    let result_lower = client.where_cond(active, &clipped_lower, lower)?;
    let result_upper = client.where_cond(active, &clipped_upper, upper)?;
    check_finite(client, &result_lower)?;
    check_finite(client, &result_upper)?;
    reject(client, &client.gt(&result_lower, &result_upper)?)?;
    ctx.bounds(&result_lower, &result_upper)
}

fn reject<R, C>(client: &C, mask: &Tensor<R>) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let any = client.any(mask, &[], false)?;
    let invalid = match mask.dtype() {
        DType::F32 => any.item::<f32>()? != 0.0,
        DType::F64 => any.item::<f64>()? != 0.0,
        _ => return Err(ArithmeticError::ControlMaskDType),
    };
    if invalid {
        return Err(ArithmeticError::InvalidBounds);
    }
    Ok(())
}

fn check_not_nan<R, C>(client: &C, value: &Tensor<R>) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    reject(client, &client.cast(&client.isnan(value)?, value.dtype())?)
}

fn check_finite<R, C>(client: &C, value: &Tensor<R>) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    check_not_nan(client, value)?;
    reject(client, &client.cast(&client.isinf(value)?, value.dtype())?)
}

fn check_nonnegative_extended<R, C>(
    client: &C,
    value: &Tensor<R>,
    zero: &Tensor<R>,
) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R>,
{
    check_not_nan(client, value)?;
    reject(client, &client.lt(value, zero)?)
}
