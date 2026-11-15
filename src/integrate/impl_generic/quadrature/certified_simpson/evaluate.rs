//! Callback admission before their values enter certified arithmetic.

use super::queries::Queries;
use super::{admission, queries};
use crate::DType;
use crate::common::arithmetic::interval::Interval;
use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::traits::{
    CertifiedSimpsonPanel, CertifiedSimpsonQuery, CertifiedSimpsonValues,
};
use numr::ops::{CompareOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) struct Evaluated<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub bounds: Interval<R>,
    pub fourth: Tensor<R>,
}

pub(super) fn callbacks<R, C, F, D>(
    client: &C,
    queries: &Queries<R>,
    mut values: F,
    mut fourth_bound: D,
) -> IntegrateResult<Evaluated<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
    F: FnMut(&CertifiedSimpsonQuery<R>) -> IntegrateResult<CertifiedSimpsonValues<R>>,
    D: FnMut(&CertifiedSimpsonPanel<R>) -> IntegrateResult<Tensor<R>>,
{
    let dtype = queries.values.point.dtype();
    let output = values(&queries.values)?;
    let mut invalid = Tensor::zeros(&[1], dtype, client.device())?;
    for (name, value) in [
        ("callback value", &output.value),
        ("callback lower", &output.lower),
        ("callback upper", &output.upper),
    ] {
        admission::tensor(client, value, queries.plan.value_points, dtype, name)?;
        let reduced = client.any(&admission::nonfinite(client, value)?, &[], false)?;
        invalid = client.maximum(&invalid, &reduced)?;
    }
    let inverted = client.maximum(
        &client.gt(&output.lower, &output.value)?,
        &client.gt(&output.value, &output.upper)?,
    )?;
    invalid = client.maximum(&invalid, &client.any(&inverted, &[], false)?)?;
    if admission::any(client, &invalid)? {
        return Err(IntegrateError::InvalidInput {
            context: "Certified Simpson callback values contain nonfinite or inverted bounds. Return finite bounds containing each nominal value.".into(),
        });
    }
    // Later callbacks can retain aliases to the returned value tensors.
    let shape = [queries.plan.active, queries.plan.nodes];
    let value = queries::owned(client, &output.value)?.reshape(&shape)?;
    let lower = queries::owned(client, &output.lower)?.reshape(&shape)?;
    let upper = queries::owned(client, &output.upper)?.reshape(&shape)?;
    let fourth = fourth_bound(&queries.panels)?;
    admission::tensor(
        client,
        &fourth,
        queries.plan.panel_points,
        dtype,
        "fourth derivative bound",
    )?;
    let zero = Tensor::zeros(&[1], dtype, client.device())?;
    let invalid = client.maximum(
        &admission::nonfinite(client, &fourth)?,
        &client.lt(&fourth, &zero)?,
    )?;
    if admission::any(client, &invalid)? {
        return Err(IntegrateError::InvalidInput {
            context: "Certified Simpson fourth-derivative bounds contain nonfinite or negative values. Return finite nonnegative bounds.".into(),
        });
    }
    let panels = [queries.plan.active, queries.plan.panels];
    Ok(Evaluated {
        value,
        bounds: Interval { lower, upper },
        fourth: queries::owned(client, &fourth)?.reshape(&panels)?,
    })
}
