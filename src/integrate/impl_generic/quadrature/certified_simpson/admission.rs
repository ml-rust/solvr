//! Metadata, input, and resource admission before callback allocation.

use crate::DType;
use crate::common::arithmetic::envelope;
use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::traits::CertifiedSimpsonOptions;
use numr::ops::{CompareOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Device, FloatArithmeticContract, Runtime, RuntimeClient};
use numr::tensor::Tensor;

#[derive(Clone, Copy)]
pub(super) struct QueryPlan {
    pub batch: usize,
    pub active: usize,
    pub panels: usize,
    pub nodes: usize,
    pub value_points: usize,
    pub panel_points: usize,
}

fn parameter(name: &str, message: String) -> IntegrateError {
    IntegrateError::InvalidParameter {
        parameter: name.into(),
        message,
    }
}

pub(super) fn count(value: usize, dtype: DType, name: &str) -> IntegrateResult<()> {
    u32::try_from(value).map_err(|_| parameter(name, format!(
        "Certified Simpson {name}={value} exceeds u32::MAX. Reduce the batch or panel count."
    )))?;
    let bytes = value.checked_mul(dtype.size_in_bytes()).ok_or_else(|| parameter(name, format!(
        "Certified Simpson {name}={value} overflows allocation bytes. Reduce the batch or panel count."
    )))?;
    if bytes > isize::MAX as usize {
        return Err(parameter(
            name,
            format!(
                "Certified Simpson {name}={value} exceeds addressable allocation bytes. Reduce the batch or panel count."
            ),
        ));
    }
    Ok(())
}

pub(super) fn metadata<R, C>(
    client: &C,
    left: &Tensor<R>,
    right: &Tensor<R>,
    options: CertifiedSimpsonOptions,
) -> IntegrateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + TwoSumOps<R> + RuntimeClient<R>,
{
    if !options.panels.is_power_of_two() {
        return Err(parameter(
            "panels",
            format!(
                "Certified Simpson panels={} requires a positive power of two. Select a power-of-two panel count.",
                options.panels
            ),
        ));
    }
    if options.max_points == 0 {
        return Err(parameter(
            "max_points",
            "Certified Simpson max_points=0 forbids every query. Supply a positive query budget."
                .into(),
        ));
    }
    if !matches!(left.dtype(), DType::F32 | DType::F64) {
        return Err(parameter(
            "dtype",
            format!(
                "Certified Simpson dtype={:?} requires F32 or F64. Select a supported dtype.",
                left.dtype()
            ),
        ));
    }
    tensor(client, left, left.numel(), left.dtype(), "left")?;
    tensor(client, right, left.numel(), left.dtype(), "right")?;
    count(left.numel(), left.dtype(), "batch")?;
    count(left.numel(), DType::I64, "owner count")?;
    let intervals = options.panels.checked_mul(2).ok_or_else(|| {
        parameter(
            "panels",
            format!(
                "Certified Simpson panels={} overflows its node count. Reduce panels.",
                options.panels
            ),
        )
    })?;
    let nodes = intervals.checked_add(1).ok_or_else(|| {
        parameter(
            "panels",
            format!(
                "Certified Simpson panels={} overflows its node count. Reduce panels.",
                options.panels
            ),
        )
    })?;
    count(nodes, left.dtype(), "node count")?;
    count(nodes, DType::I64, "node indices")?;
    if left.dtype() == DType::F32 && intervals > (1usize << 24) {
        return Err(parameter(
            "panels",
            format!(
                "Certified Simpson F32 panels={} exceeds the exact dyadic grid limit 2^23. Reduce panels.",
                options.panels
            ),
        ));
    }
    if R::float_arithmetic_contract(left.dtype()) != FloatArithmeticContract::Ieee754Basic {
        return Err(IntegrateError::UnsupportedArithmetic {
            dtype: left.dtype(),
            failed_probes: vec!["provider_contract"],
        });
    }
    envelope::check(client, left.dtype())?;
    Ok(())
}

pub(super) fn tensor<R, C>(
    client: &C,
    value: &Tensor<R>,
    len: usize,
    dtype: DType,
    name: &str,
) -> IntegrateResult<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if value.ndim() != 1
        || value.numel() != len
        || value.dtype() != dtype
        || !client.device().is_same(value.device())
    {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "Certified Simpson {name} has shape {:?}, dtype {:?}, or a foreign device. Supply [{len}] {dtype:?} on the configured client.",
                value.shape(),
                value.dtype()
            ),
        });
    }
    Ok(())
}

pub(super) fn any<R, C>(client: &C, mask: &Tensor<R>) -> IntegrateResult<bool>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    if mask.numel() == 0 {
        return Ok(false);
    }
    let value = client.any(mask, &[], false)?;
    match mask.dtype() {
        DType::F32 => Ok(value.item::<f32>()? != 0.0),
        DType::F64 => Ok(value.item::<f64>()? != 0.0),
        _ => Err(IntegrateError::NumericalError {
            message:
                "Certified Simpson control mask requires F32 or F64. Select a supported dtype."
                    .into(),
        }),
    }
}

pub(super) fn nonfinite<R, C>(client: &C, value: &Tensor<R>) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    Ok(client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?)
}

pub(super) fn endpoints<R, C>(
    client: &C,
    left: &Tensor<R>,
    right: &Tensor<R>,
) -> IntegrateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R>,
{
    let invalid = client.maximum(
        &client.maximum(&nonfinite(client, left)?, &nonfinite(client, right)?)?,
        &client.gt(left, right)?,
    )?;
    if any(client, &invalid)? {
        return Err(IntegrateError::InvalidInput {
            context: "Certified Simpson endpoints contain nonfinite values or left > right. Supply finite ordered endpoints.".into(),
        });
    }
    Ok(())
}

pub(super) fn plan(
    batch: usize,
    active: usize,
    dtype: DType,
    options: CertifiedSimpsonOptions,
) -> IntegrateResult<QueryPlan> {
    let panels = options.panels;
    let nodes = panels.checked_mul(2).and_then(|n| n.checked_add(1));
    let value_points = nodes.and_then(|n| n.checked_mul(active));
    let panel_points = panels.checked_mul(active);
    let combined = value_points.and_then(|n| panel_points.and_then(|m| n.checked_add(m)));
    let (nodes, value_points, panel_points, combined) = match (
        nodes,
        value_points,
        panel_points,
        combined,
    ) {
        (Some(n), Some(v), Some(p), Some(c)) => (n, v, p, c),
        _ => {
            return Err(parameter(
                "max_points",
                format!(
                    "Certified Simpson active={active}, panels={panels} overflows query counts. Reduce the batch or panel count."
                ),
            ));
        }
    };
    if combined > options.max_points {
        return Err(parameter(
            "max_points",
            format!(
                "Certified Simpson needs {combined} queries but max_points={}. Increase the budget or reduce panels.",
                options.max_points
            ),
        ));
    }
    for (name, n) in [
        ("value queries", value_points),
        ("derivative queries", panel_points),
    ] {
        count(n, dtype, name)?;
        count(n, DType::I64, name)?;
    }
    Ok(QueryPlan {
        batch,
        active,
        panels,
        nodes,
        value_points,
        panel_points,
    })
}
