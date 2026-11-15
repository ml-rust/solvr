//! Fixed pairwise Simpson assembly and independent batch receipt storage.

use super::evaluate::Evaluated;
use super::queries::Queries;
use crate::DType;
use crate::common::arithmetic::interval::IntervalArithmetic;
use crate::common::arithmetic::reduction::{pairwise_interval_sum_rows, pairwise_sum_rows};
use crate::integrate::error::IntegrateResult;
use crate::integrate::traits::CertifiedSimpsonResult;
use numr::ops::{CompareOps, NextRepresentableOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn zeros<R, C>(
    client: &C,
    batch: usize,
    dtype: DType,
) -> IntegrateResult<CertifiedSimpsonResult<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    Ok(CertifiedSimpsonResult {
        value: Tensor::zeros(&[batch], dtype, client.device())?,
        lower: Tensor::zeros(&[batch], dtype, client.device())?,
        upper: Tensor::zeros(&[batch], dtype, client.device())?,
        abs_error_bound: Tensor::zeros(&[batch], dtype, client.device())?,
        quadrature_error_bound: Tensor::zeros(&[batch], dtype, client.device())?,
    })
}

fn weights<R, C>(client: &C, panels: usize, dtype: DType) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let four = Tensor::full_scalar(&[panels], dtype, 4.0, client.device())?;
    let two = Tensor::full_scalar(&[panels], dtype, 2.0, client.device())?;
    let alternating = client.stack(&[&four, &two], 1)?.reshape(&[2 * panels])?;
    let one = Tensor::ones(&[1], dtype, client.device())?;
    // Drop the final interior weight and append the exact endpoint weight.
    let interior = alternating.narrow(0, 0, 2 * panels - 1)?;
    Ok(client.cat(&[&one, &interior, &one], 0)?.unsqueeze(0)?)
}

pub(super) fn finish<R, C>(
    client: &C,
    arithmetic: &mut IntervalArithmetic<'_, R, C>,
    queries: Queries<R>,
    evaluated: Evaluated<R>,
) -> IntegrateResult<CertifiedSimpsonResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    let plan = queries.plan;
    let dtype = evaluated.value.dtype();
    let weight = weights(client, plan.panels, dtype)?;
    let weighted = client.mul(&evaluated.value, &weight)?;
    arithmetic.finite(&weighted)?;
    let weighted_sum = pairwise_sum_rows(arithmetic, &weighted)?;
    let weight_interval = arithmetic.point(&weight)?;
    let weighted_bounds = arithmetic.mul(&evaluated.bounds, &weight_interval)?;
    let sum_bounds = pairwise_interval_sum_rows(arithmetic, &weighted_bounds)?;

    let divisor = Tensor::full_scalar(&[1], dtype, 6.0 * plan.panels as f64, client.device())?;
    let scale = client.div(&queries.width, &divisor)?;
    let value = client.mul(&weighted_sum, &scale)?;
    arithmetic.finite(&scale)?;
    arithmetic.finite(&value)?;
    let scale_bounds = arithmetic.div_positive(&queries.span, &divisor)?;
    let simpson = arithmetic.mul(&sum_bounds, &scale_bounds)?;
    arithmetic.check("Simpson weighted sum")?;

    let remainder = remainder(client, arithmetic, &queries, &evaluated.fourth)?;
    let error_interval = arithmetic.bounds(client.neg(&remainder)?, remainder.clone())?;
    let integral = arithmetic.add(&simpson, &error_interval)?;
    let nominal = arithmetic.point(&value)?;
    let lower_point = arithmetic.point(&integral.lower)?;
    let upper_point = arithmetic.point(&integral.upper)?;
    let below = arithmetic.sub(&nominal, &lower_point)?;
    let above = arithmetic.sub(&upper_point, &nominal)?;
    let zero = Tensor::zeros(&[1], dtype, client.device())?;
    let absolute = client.maximum(&zero, &client.maximum(&below.upper, &above.upper)?)?;
    arithmetic.finite(&absolute)?;
    arithmetic.check("integral receipt")?;

    let empty = Tensor::zeros(&[plan.batch], dtype, client.device())?;
    Ok(CertifiedSimpsonResult {
        value: client.scatter(&empty, 0, &queries.owners, &value)?,
        lower: client.scatter(&empty, 0, &queries.owners, &integral.lower)?,
        upper: client.scatter(&empty, 0, &queries.owners, &integral.upper)?,
        abs_error_bound: client.scatter(&empty, 0, &queries.owners, &absolute)?,
        quadrature_error_bound: client.scatter(&empty, 0, &queries.owners, &remainder)?,
    })
}

fn remainder<R, C>(
    client: &C,
    arithmetic: &mut IntervalArithmetic<'_, R, C>,
    queries: &Queries<R>,
    fourth: &Tensor<R>,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    let dtype = fourth.dtype();
    let zero = Tensor::zeros(&[1], dtype, client.device())?;
    let panels = Tensor::full_scalar(&[1], dtype, queries.plan.panels as f64, client.device())?;
    let required = client.gt(fourth, &zero)?;
    // Zero fourth derivatives suppress unused width arithmetic and powers.
    let used_span = arithmetic.bounds(
        client.where_cond(&required, &queries.span.lower.unsqueeze(1)?, &zero)?,
        client.where_cond(&required, &queries.span.upper.unsqueeze(1)?, &zero)?,
    )?;
    let delta = arithmetic.div_positive(&used_span, &panels)?;
    let squared = arithmetic.mul(&delta, &delta)?;
    let fourth_power = arithmetic.mul(&squared, &squared)?;
    let fifth_power = arithmetic.mul(&fourth_power, &delta)?;
    let derivative = arithmetic.point(fourth)?;
    let numerator = arithmetic.mul(&fifth_power, &derivative)?;
    let denominator = Tensor::full_scalar(&[1], dtype, 2880.0, client.device())?;
    let panel_error = arithmetic.div_positive(&numerator, &denominator)?;
    let nonnegative = client.maximum(&panel_error.upper, &zero)?;
    let represented_upper = arithmetic.point(&nonnegative)?;
    let total = pairwise_interval_sum_rows(arithmetic, &represented_upper)?;
    let result = client.maximum(&total.upper, &zero)?;
    arithmetic.finite(&result)?;
    arithmetic.check("Simpson fourth-derivative remainder")?;
    Ok(result)
}
