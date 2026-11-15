//! Runtime-generic Newton corrections with independent frozen lanes.

use crate::DType;
use crate::optimize::roots::{BatchedNewtonOptions, BatchedNewtonResult};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::ops::{CompareOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Evaluate batched corrections without transferring individual lanes to the host.
pub fn newton_batched_impl<R, C, F>(
    client: &C,
    correction_callback: F,
    initial: &Tensor<R>,
    options: &BatchedNewtonOptions,
) -> OptimizeResult<BatchedNewtonResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
{
    if initial.numel() == 0 || !matches!(initial.dtype(), DType::F32 | DType::F64) {
        return Err(input_error(&format!(
            "initial dtype={:?} shape={:?}: require a nonempty F32 or F64 tensor",
            initial.dtype(),
            initial.shape()
        )));
    }
    if !initial.device().is_same(client.device()) {
        return Err(input_error(&format!(
            "initial device_id={}, configured client device_id={}: require matching devices",
            initial.device().id(),
            client.device().id()
        )));
    }
    let tolerance = options.correction_tolerance;
    if !tolerance.is_finite()
        || tolerance <= 0.0
        || (initial.dtype() == DType::F32
            && (!(tolerance as f32).is_finite() || tolerance as f32 <= 0.0))
    {
        return Err(input_error(&format!(
            "correction_tolerance={tolerance} dtype={:?}: require a positive finite value representable in the input dtype",
            initial.dtype()
        )));
    }
    if options.max_iter == 0 {
        return Err(input_error(&format!(
            "max_iter={}: require a positive iteration limit",
            options.max_iter
        )));
    }
    if any(client, &nonfinite_mask(client, initial)?)? {
        return Err(input_error(&format!(
            "initial dtype={:?} shape={:?} contains nonfinite values: provide finite iterates",
            initial.dtype(),
            initial.shape()
        )));
    }
    let zero = client.fill(initial.shape(), 0.0, initial.dtype())?;
    let one = client.fill(initial.shape(), 1.0, initial.dtype())?;
    let threshold = client.fill(initial.shape(), tolerance, initial.dtype())?;
    let mut x = initial.clone();
    let mut correction = zero.clone();
    let mut converged = zero.clone();
    let mut nonfinite = zero;
    let mut active = one.clone();
    let mut iterations = 0;
    for _ in 0..options.max_iter {
        let evaluated = correction_callback(&x)?;
        if evaluated.shape() != initial.shape()
            || evaluated.dtype() != initial.dtype()
            || !evaluated.device().is_same(initial.device())
        {
            return Err(input_error(&format!(
                "callback dtype={:?} shape={:?} device_id={}, initial dtype={:?} shape={:?} device_id={}: require matching metadata",
                evaluated.dtype(),
                evaluated.shape(),
                evaluated.device().id(),
                initial.dtype(),
                initial.shape(),
                initial.device().id()
            )));
        }
        correction = client.where_cond(&active, &evaluated, &correction)?;
        let invalid_correction = nonfinite_mask(client, &evaluated)?;
        let finite_correction = client.sub(&one, &invalid_correction)?;
        let newly_converged = client.mul(
            &active,
            &client.mul(
                &finite_correction,
                &client.lt(&client.abs(&evaluated)?, &threshold)?,
            )?,
        )?;
        converged = client.maximum(&converged, &newly_converged)?;
        let updating = client.mul(&active, &client.sub(&one, &newly_converged)?)?;
        let candidate = client.sub(&x, &evaluated)?;
        let invalid = client.maximum(&invalid_correction, &nonfinite_mask(client, &candidate)?)?;
        nonfinite = client.maximum(&nonfinite, &client.mul(&updating, &invalid)?)?;
        active = client.mul(&updating, &client.sub(&one, &invalid)?)?;
        x = client.where_cond(&active, &candidate, &x)?;
        iterations += 1;
        // Only the global active mask crosses the device boundary each round.
        if !any(client, &active)? {
            break;
        }
    }
    Ok(BatchedNewtonResult {
        x,
        correction,
        converged,
        nonfinite,
        iterations,
    })
}

fn nonfinite_mask<R, C>(client: &C, value: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    Ok(client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?)
}

fn any<R, C>(client: &C, mask: &Tensor<R>) -> OptimizeResult<bool>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R>,
{
    let value = client.any(mask, &[], false)?;
    match mask.dtype() {
        DType::F32 => Ok(value.item::<f32>()? != 0.0),
        DType::F64 => Ok(value.item::<f64>()? != 0.0),
        _ => Err(input_error("control masks require F32 or F64")),
    }
}

fn input_error(context: &str) -> OptimizeError {
    OptimizeError::InvalidInput {
        context: format!("newton_batched: {context}"),
    }
}
