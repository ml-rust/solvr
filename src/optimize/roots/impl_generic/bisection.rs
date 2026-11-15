//! Runtime-generic batched bisection and upper-bracket expansion.

use crate::DType;
use crate::optimize::roots::{BatchedBisectionOptions, BatchedBracketResult};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::ops::{CompareOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Solve independent scalar roots without transferring lane data to the host.
pub fn bisect_batched_impl<R, C, F>(
    client: &C,
    f: F,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    options: &BatchedBisectionOptions,
) -> OptimizeResult<BatchedBracketResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
{
    check_options(options)?;
    if !matches!(lower.dtype(), DType::F32 | DType::F64) || lower.numel() == 0 {
        return Err(input_error("bounds require nonempty F32 or F64 tensors"));
    }
    check_metadata(upper, lower)?;
    if !lower.device().is_same(client.device()) {
        return Err(input_error(
            "bounds and configured client require the same device",
        ));
    }
    let zero = client.fill(lower.shape(), 0.0, lower.dtype())?;
    let all = client.eq(&zero, &zero)?;
    check_finite(client, lower, &all, "lower bound")?;
    check_finite(client, upper, &all, "upper bound")?;
    if any(client, &client.gt(lower, upper)?)? {
        return Err(input_error("bounds require lower <= upper in every lane"));
    }

    let mut lower = lower.clone();
    let mut upper = upper.clone();
    let mut f_lower = f(&lower)?;
    check_metadata(&f_lower, &lower)?;
    check_finite(client, &f_lower, &all, "lower residual")?;
    let lower_zero = client.eq(&f_lower, &zero)?;
    if any(
        client,
        &client.mul(
            &client.eq(&lower, &upper)?,
            &client.rsub_scalar(&lower_zero, 1.0)?,
        )?,
    )? {
        return Err(input_error(
            "degenerate intervals require an exact zero residual",
        ));
    }
    upper = client.where_cond(&lower_zero, &lower, &upper)?;
    let mut f_upper = f(&upper)?;
    check_metadata(&f_upper, &lower)?;
    check_finite(
        client,
        &f_upper,
        &client.rsub_scalar(&lower_zero, 1.0)?,
        "upper residual",
    )?;
    f_upper = client.where_cond(&lower_zero, &zero, &f_upper)?;
    collapse_exact(
        client,
        &mut lower,
        &mut upper,
        &mut f_lower,
        &mut f_upper,
        &zero,
    )?;

    let mut expansions = 0;
    loop {
        let same_sign = same_sign_nonzero(client, &f_lower, &f_upper, &zero)?;
        if !any(client, &same_sign)? {
            break;
        }
        if expansions == options.max_expansions {
            return Err(input_error(
                "same-sign residuals remain after max_expansions, provide a wider bracket or increase max_expansions",
            ));
        }
        let width = client.sub(&upper, &lower)?;
        check_finite(client, &width, &same_sign, "expansion width")?;
        let candidate = client.add(
            &lower,
            &client.mul_scalar(&width, options.expansion_factor)?,
        )?;
        check_finite(client, &candidate, &same_sign, "expanded upper bound")?;
        if any(
            client,
            &client.mul(&same_sign, &client.le(&candidate, &upper)?)?,
        )? {
            return Err(numerical_error(
                "upper-bound expansion stagnates, increase expansion_factor or change bounds",
            ));
        }
        upper = client.where_cond(&same_sign, &candidate, &upper)?;
        let values = f(&upper)?;
        check_metadata(&values, &lower)?;
        check_finite(client, &values, &same_sign, "expanded upper residual")?;
        f_upper = client.where_cond(&same_sign, &values, &f_upper)?;
        collapse_exact(
            client,
            &mut lower,
            &mut upper,
            &mut f_lower,
            &mut f_upper,
            &zero,
        )?;
        expansions += 1;
    }

    let mut active = all;
    for iterations in 0..=options.max_iter {
        let width = client.sub(&upper, &lower)?;
        check_finite(client, &width, &active, "bracket width")?;
        let scale = client.maximum(&client.abs(&lower)?, &client.abs(&upper)?)?;
        let tolerance =
            client.add_scalar(&client.mul_scalar(&scale, options.rtol)?, options.atol)?;
        check_finite(client, &tolerance, &active, "width tolerance")?;
        active = client.mul(&active, &client.gt(&width, &tolerance)?)?;
        if !any(client, &active)? {
            return Ok(BatchedBracketResult {
                lower,
                upper,
                f_lower,
                f_upper,
                iterations,
                expansions,
            });
        }
        if iterations == options.max_iter {
            return Err(OptimizeError::DidNotConverge {
                iterations,
                tolerance: options.atol,
                context:
                    "bisect_batched: active brackets exceed width tolerance, increase max_iter"
                        .into(),
            });
        }
        let midpoint = client.add(&lower, &client.mul_scalar(&width, 0.5)?)?;
        let stagnant = client.maximum(
            &client.le(&midpoint, &lower)?,
            &client.ge(&midpoint, &upper)?,
        )?;
        if any(client, &client.mul(&active, &stagnant)?)? {
            return Err(numerical_error(
                "midpoint stagnates before width tolerance, increase atol or rtol",
            ));
        }
        // Frozen lanes retain their lower endpoint during callback evaluation.
        let midpoint = client.where_cond(&active, &midpoint, &lower)?;
        let values = f(&midpoint)?;
        check_metadata(&values, &lower)?;
        check_finite(client, &values, &active, "midpoint residual")?;
        let exact = client.mul(&active, &client.eq(&values, &zero)?)?;
        let same_lower = client.eq(&client.lt(&values, &zero)?, &client.lt(&f_lower, &zero)?)?;
        let move_lower = client.mul(&active, &client.maximum(&same_lower, &exact)?)?;
        let move_upper = client.mul(
            &active,
            &client.maximum(&client.rsub_scalar(&same_lower, 1.0)?, &exact)?,
        )?;
        lower = client.where_cond(&move_lower, &midpoint, &lower)?;
        f_lower = client.where_cond(&move_lower, &values, &f_lower)?;
        upper = client.where_cond(&move_upper, &midpoint, &upper)?;
        f_upper = client.where_cond(&move_upper, &values, &f_upper)?;
    }
    Err(numerical_error("bisection exhausted its iteration range"))
}

fn check_options(options: &BatchedBisectionOptions) -> OptimizeResult<()> {
    for (name, value) in [("atol", options.atol), ("rtol", options.rtol)] {
        if !value.is_finite() || value < 0.0 {
            return Err(OptimizeError::InvalidParameter {
                parameter: name.into(),
                message: format!("received {value}, use a finite nonnegative tolerance"),
            });
        }
    }
    if options.atol == 0.0 && options.rtol == 0.0 {
        return Err(input_error(
            "atol and rtol are both zero, provide a positive tolerance",
        ));
    }
    if options.max_iter == 0 {
        return Err(OptimizeError::InvalidParameter {
            parameter: "max_iter".into(),
            message: format!(
                "received {}, use a positive iteration limit",
                options.max_iter
            ),
        });
    }
    if !options.expansion_factor.is_finite() || options.expansion_factor <= 1.0 {
        return Err(OptimizeError::InvalidParameter {
            parameter: "expansion_factor".into(),
            message: format!(
                "received {}, use a finite value greater than one",
                options.expansion_factor
            ),
        });
    }
    Ok(())
}

fn check_metadata<R: Runtime<DType = DType>>(
    value: &Tensor<R>,
    bounds: &Tensor<R>,
) -> OptimizeResult<()> {
    if value.shape() != bounds.shape()
        || value.dtype() != bounds.dtype()
        || !value.device().is_same(bounds.device())
    {
        return Err(input_error(
            "tensor shape, dtype, and device must match the lower bound",
        ));
    }
    Ok(())
}

fn any<R, C>(client: &C, mask: &Tensor<R>) -> OptimizeResult<bool>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R>,
{
    // Only aggregate control-flow scalars cross the device boundary.
    let reduced = client.any(mask, &[], false)?;
    match mask.dtype() {
        DType::F32 => Ok(reduced.item::<f32>()? != 0.0),
        DType::F64 => Ok(reduced.item::<f64>()? != 0.0),
        _ => Err(numerical_error(
            "unsupported control mask dtype, use F32 or F64 bounds",
        )),
    }
}

fn check_finite<R, C>(
    client: &C,
    value: &Tensor<R>,
    active: &Tensor<R>,
    name: &str,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let nan = client.cast(&client.isnan(value)?, value.dtype())?;
    let infinite = client.cast(&client.isinf(value)?, value.dtype())?;
    let invalid = client.maximum(&nan, &infinite)?;
    if any(client, &client.mul(active, &invalid)?)? {
        return Err(numerical_error(&format!(
            "nonfinite {name} in an active lane, use finite bounds and residuals"
        )));
    }
    Ok(())
}

fn same_sign_nonzero<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    zero: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: CompareOps<R>,
{
    let nonzero = client.mul(&client.ne(a, zero)?, &client.ne(b, zero)?)?;
    let same = client.eq(&client.lt(a, zero)?, &client.lt(b, zero)?)?;
    Ok(client.mul(&nonzero, &same)?)
}

fn collapse_exact<R, C>(
    client: &C,
    lower: &mut Tensor<R>,
    upper: &mut Tensor<R>,
    f_lower: &mut Tensor<R>,
    f_upper: &mut Tensor<R>,
    zero: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R>,
{
    let at_lower = client.eq(f_lower, zero)?;
    let at_upper = client.eq(f_upper, zero)?;
    *upper = client.where_cond(&at_lower, lower, upper)?;
    *f_upper = client.where_cond(&at_lower, zero, f_upper)?;
    *lower = client.where_cond(&at_upper, upper, lower)?;
    *f_lower = client.where_cond(&at_upper, zero, f_lower)?;
    Ok(())
}

fn input_error(context: &str) -> OptimizeError {
    OptimizeError::InvalidInput {
        context: format!("bisect_batched: {context}"),
    }
}

fn numerical_error(message: &str) -> OptimizeError {
    OptimizeError::NumericalError {
        message: format!("bisect_batched: {message}"),
    }
}
