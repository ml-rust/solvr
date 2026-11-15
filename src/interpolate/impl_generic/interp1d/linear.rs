//! Linear interpolation without denominator perturbations.

use super::validation::{finite, nonzero};
use crate::DType;
use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(in crate::interpolate::impl_generic) struct LinearEvaluation<R: Runtime<DType = DType>> {
    pub values: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub x0: Tensor<R>,
    pub x1: Tensor<R>,
    pub left: Tensor<R>,
    pub right: Tensor<R>,
    pub width: Tensor<R>,
    pub offset: Tensor<R>,
    pub difference: Tensor<R>,
    pub coordinate: Tensor<R>,
    pub increment: Tensor<R>,
    pub at_lower: Tensor<R>,
    pub at_upper: Tensor<R>,
    pub constant: Tensor<R>,
    pub active: Tensor<R>,
}

pub(super) fn evaluate<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    Ok(evaluate_captured(client, x, y, query)?.values)
}

pub(in crate::interpolate::impl_generic) fn evaluate_captured<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<LinearEvaluation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let indices = client.searchsorted(x, query, false)?;
    let n = x.numel();
    let max_index = match indices.dtype() {
        DType::I32 => i32::MAX as u128,
        DType::I64 => 1u128 << 53,
        dtype => {
            return Err(InterpolateError::InvalidParameter {
                parameter: "searchsorted indices".into(),
                message: format!("dtype={dtype:?}: require I32 or I64"),
            });
        }
    };
    if n as u128 > max_index {
        return Err(InterpolateError::InvalidParameter {
            parameter: "grid length".into(),
            message: format!("n={n}: exceeds exact portable index capacity {max_index}"),
        });
    }
    let one_index = Tensor::ones(indices.shape(), indices.dtype(), client.device())?;
    let last_index = Tensor::full_scalar(
        indices.shape(),
        indices.dtype(),
        (n - 1) as f64,
        client.device(),
    )?;
    // Clamp interval indices only. Query coordinates retain their extrapolation distance.
    let upper = client.maximum(&client.minimum(&indices, &last_index)?, &one_index)?;
    let lower = client.sub(&upper, &one_index)?;
    let x0 = client.index_select(x, 0, &lower)?;
    let x1 = client.index_select(x, 0, &upper)?;
    let y0 = client.index_select(y, 0, &lower)?;
    let y1 = client.index_select(y, 0, &upper)?;
    let at_left = client.eq(query, &x0)?;
    let at_right = client.eq(query, &x1)?;
    let zero = Tensor::zeros(query.shape(), query.dtype(), client.device())?;
    let one = Tensor::ones(query.shape(), query.dtype(), client.device())?;
    let knots = client.maximum(&at_left, &at_right)?;
    let constant = client.eq(&y0, &y1)?;
    let active = client.sub(&one, &client.maximum(&knots, &constant)?)?;
    // Knots and constant intervals bypass coordinate arithmetic that cannot affect their stored value.
    let left = client.where_cond(&active, &y0, &zero)?;
    let right = client.where_cond(&active, &y1, &zero)?;
    let dy = client.sub(&right, &left)?;
    let offset = client.sub(
        &client.where_cond(&active, query, &zero)?,
        &client.where_cond(&active, &x0, &zero)?,
    )?;
    let dx = client.where_cond(&active, &client.sub(&x1, &x0)?, &one)?;
    finite(client, &dy, "active value difference")?;
    finite(client, &offset, "active query offset")?;
    nonzero(
        client,
        &dy,
        &client.ne(&left, &right)?,
        "active value difference",
    )?;
    nonzero(client, &offset, &active, "active query offset")?;
    let t = client.div(&offset, &dx)?;
    finite(client, &t, "linear coordinate")?;
    nonzero(client, &t, &active, "linear coordinate")?;
    let scaled = client.mul(&dy, &t)?;
    finite(client, &scaled, "linear value increment")?;
    nonzero(
        client,
        &scaled,
        &client.mul(&active, &client.ne(&dy, &zero)?)?,
        "linear value increment",
    )?;
    let value = client.add(&left, &scaled)?;
    finite(client, &value, "linear interpolated value")?;
    let value = client.where_cond(&constant, &y0, &value)?;
    let value = client.where_cond(&at_right, &y1, &value)?;
    let values = client.where_cond(&at_left, &y0, &value)?;
    Ok(LinearEvaluation {
        values,
        lower,
        upper,
        x0,
        x1,
        left,
        right,
        width: dx,
        offset,
        difference: dy,
        coordinate: t,
        increment: scaled,
        at_lower: at_left,
        at_upper: at_right,
        constant,
        active,
    })
}
