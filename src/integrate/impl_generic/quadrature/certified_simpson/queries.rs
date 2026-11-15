//! Native exact-grid coordinate enclosures and original-owner queries.

use super::admission::QueryPlan;
use crate::DType;
use crate::common::arithmetic::interval::{Interval, IntervalArithmetic};
use crate::integrate::error::IntegrateResult;
use crate::integrate::traits::{CertifiedSimpsonPanel, CertifiedSimpsonQuery};
use numr::ops::{CompareOps, NextRepresentableOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) struct Queries<R: Runtime<DType = DType>> {
    pub owners: Tensor<R>,
    pub values: CertifiedSimpsonQuery<R>,
    pub panels: CertifiedSimpsonPanel<R>,
    pub width: Tensor<R>,
    pub span: Interval<R>,
    pub plan: QueryPlan,
}

/// Materialize through the configured client without changing signed zero.
pub(super) fn owned<R, C>(client: &C, value: &Tensor<R>) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let output = Tensor::empty(value.shape(), value.dtype(), client.device())?;
    client.copy_into(&output, value)?;
    Ok(output)
}

fn owners<R, C>(client: &C, original: &Tensor<R>, width: usize) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let expanded = original
        .unsqueeze(1)?
        .broadcast_to(&[original.numel(), width])?;
    let zero = Tensor::zeros(&[1], DType::I64, client.device())?;
    Ok(client
        .add(&expanded, &zero)?
        .reshape(&[original.numel() * width])?)
}

pub(super) fn create<R, C>(
    client: &C,
    arithmetic: &mut IntervalArithmetic<'_, R, C>,
    left: &Tensor<R>,
    right: &Tensor<R>,
    original: Tensor<R>,
    plan: QueryPlan,
) -> IntegrateResult<Queries<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    let dtype = left.dtype();
    let a = client.index_select(left, 0, &original)?.unsqueeze(1)?;
    let b = client.index_select(right, 0, &original)?.unsqueeze(1)?;
    let ai = arithmetic.point(&a)?;
    let bi = arithmetic.point(&b)?;
    let width = client.sub(&b, &a)?;
    arithmetic.finite(&width)?;
    let difference = arithmetic.sub(&bi, &ai)?;
    let zero = Tensor::zeros(&[1], dtype, client.device())?;
    let span = arithmetic.bounds(client.maximum(&difference.lower, &zero)?, difference.upper)?;
    arithmetic.check("coordinate width")?;

    let grid = client.linspace(0.0, 1.0, plan.nodes, dtype)?.unsqueeze(0)?;
    let grid_interval = arithmetic.point(&grid)?;
    let displacement = arithmetic.mul(&span, &grid_interval)?;
    let coordinates = arithmetic.add(&ai, &displacement)?;
    let represented = client.add(&a, &client.mul(&width, &grid)?)?;
    arithmetic.finite(&represented)?;
    let represented = client.minimum(&client.maximum(&represented, &a)?, &b)?;
    let lower = client.maximum(&client.minimum(&coordinates.lower, &represented)?, &a)?;
    let upper = client.minimum(&client.maximum(&coordinates.upper, &represented)?, &b)?;

    let one = Tensor::ones(&[1], dtype, client.device())?;
    let first = client.eq(&grid, &zero)?;
    let last = client.eq(&grid, &one)?;
    let endpoint = |value: &Tensor<R>| -> IntegrateResult<Tensor<R>> {
        Ok(client.where_cond(&first, &a, &client.where_cond(&last, &b, value)?)?)
    };
    let point = endpoint(&represented)?;
    let bounds = arithmetic.bounds(endpoint(&lower)?, endpoint(&upper)?)?;
    arithmetic.mark_invalid(&client.maximum(
        &client.gt(&bounds.lower, &point)?,
        &client.gt(&point, &bounds.upper)?,
    )?)?;
    arithmetic.check("coordinate enclosure")?;

    let starts = client.arange(0.0, (2 * plan.panels) as f64, 2.0, DType::I64)?;
    let ends = client.arange(2.0, (2 * plan.panels + 1) as f64, 2.0, DType::I64)?;
    let panel_lower = client.index_select(&bounds.lower, 1, &starts)?;
    let panel_upper = client.index_select(&bounds.upper, 1, &ends)?;
    let panel_bounds = arithmetic.bounds(panel_lower, panel_upper)?;
    arithmetic.check("derivative panel enclosure")?;

    Ok(Queries {
        values: CertifiedSimpsonQuery {
            owner: owners(client, &original, plan.nodes)?,
            point: point.reshape(&[plan.value_points])?,
            lower: bounds.lower.reshape(&[plan.value_points])?,
            upper: bounds.upper.reshape(&[plan.value_points])?,
        },
        panels: CertifiedSimpsonPanel {
            owner: owners(client, &original, plan.panels)?,
            lower: panel_bounds.lower.reshape(&[plan.panel_points])?,
            upper: panel_bounds.upper.reshape(&[plan.panel_points])?,
        },
        owners: original,
        width: width.reshape(&[plan.active])?,
        span: Interval {
            lower: span.lower.reshape(&[plan.active])?,
            upper: span.upper.reshape(&[plan.active])?,
        },
        plan,
    })
}
