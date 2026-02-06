//! Helper functions for DAE solver.
//!
//! This module contains shared utility functions used by the DAE solver
//! for error estimation, step control, and result building.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::DAEResultTensor;
#[cfg(feature = "sparse")]
use crate::integrate::ode::DAEVariableType;

#[cfg(feature = "sparse")]
use super::jacobian::compute_norm_scalar;

// BDF coefficients (shared with dae.rs)
pub(super) const BDF_ALPHA: [[f64; 6]; 5] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0, 0.0, 0.0],
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0, 0.0],
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
];

pub(super) const BDF_BETA: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];

pub(super) const BDF_ERROR_COEFF: [f64; 5] =
    [1.0 / 2.0, 2.0 / 9.0, 3.0 / 22.0, 12.0 / 125.0, 10.0 / 137.0];

pub(super) const SAFETY: f64 = 0.9;
pub(super) const MIN_FACTOR: f64 = 0.2;
pub(super) const MAX_FACTOR: f64 = 5.0;

/// Compute predictor using derivative information.
pub(super) fn compute_predictor_with_yp<R, C>(
    client: &C,
    y_history: &[Tensor<R>],
    yp_history: &[Tensor<R>],
    h: f64,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    if y_history.is_empty() || yp_history.is_empty() {
        return Err(IntegrateError::InvalidInput {
            context: "Empty history".to_string(),
        });
    }

    let y_n = &y_history[0];
    let yp_n = &yp_history[0];

    // First-order Taylor predictor: y_{n+1} ≈ y_n + h * yp_n
    let h_yp = client.mul_scalar(yp_n, h).map_err(to_integrate_err)?;
    client.add(y_n, &h_yp).map_err(to_integrate_err)
}

/// Compute y' from BDF formula.
pub(super) fn compute_yp_from_bdf<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_history: &[Tensor<R>],
    order: usize,
    h: f64,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let order_idx = (order - 1).min(4);
    let alpha = &BDF_ALPHA[order_idx];
    let beta = BDF_BETA[order_idx];

    let mut numerator = client
        .mul_scalar(y_new, alpha[0])
        .map_err(to_integrate_err)?;

    for i in 1..=order {
        if i <= y_history.len() {
            let term = client
                .mul_scalar(&y_history[i - 1], alpha[i])
                .map_err(to_integrate_err)?;
            numerator = client.add(&numerator, &term).map_err(to_integrate_err)?;
        }
    }

    client
        .mul_scalar(&numerator, 1.0 / (h * beta))
        .map_err(to_integrate_err)
}

/// Estimate local truncation error.
pub(super) fn estimate_error<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_pred: &Tensor<R>,
    order: usize,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let order_idx = (order - 1).min(4);
    let error_coeff = BDF_ERROR_COEFF[order_idx];
    let diff = client.sub(y_new, y_pred).map_err(to_integrate_err)?;
    client
        .mul_scalar(&diff, error_coeff)
        .map_err(to_integrate_err)
}

/// Compute normalized error (for all variables).
pub(super) fn compute_error<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    error: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let y_abs = client.abs(y_new)?;
    let y_old_abs = client.abs(y_old)?;
    let max_abs = client.maximum(&y_abs, &y_old_abs)?;
    let scale = client.add_scalar(&client.mul_scalar(&max_abs, rtol)?, atol)?;
    let normalized = client.div(error, &scale)?;
    let sq = client.mul(&normalized, &normalized)?;
    let mean_sq = client.mean(&sq, &[0], false)?;
    let rms: Vec<f64> = mean_sq.to_vec();
    Ok(rms[0].sqrt())
}

/// Compute normalized error excluding algebraic variables.
#[cfg(feature = "sparse")]
pub(super) fn compute_error_with_exclusion<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    error: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
    var_types: &Option<Vec<DAEVariableType>>,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let Some(types) = var_types else {
        return compute_error(client, y_new, error, y_old, rtol, atol);
    };

    let n = y_new.shape()[0];

    // Create mask: 1 for differential, 0 for algebraic
    let mask: Vec<f64> = types
        .iter()
        .map(|t| {
            if *t == DAEVariableType::Differential {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mask_tensor = Tensor::<R>::from_slice(&mask, &[n], client.device());

    // Mask the error
    let masked_error = client.mul(error, &mask_tensor)?;

    // Count differential variables
    let n_diff = types
        .iter()
        .filter(|t| **t == DAEVariableType::Differential)
        .count() as f64;

    if n_diff < 1.0 {
        // All algebraic - just check that error is small
        let err_norm = compute_norm_scalar(client, error, 2.0)?;
        return Ok(err_norm / atol.max(1e-10));
    }

    // Compute error using only differential variables
    let y_abs = client.abs(y_new)?;
    let y_old_abs = client.abs(y_old)?;
    let max_abs = client.maximum(&y_abs, &y_old_abs)?;
    let scale = client.add_scalar(&client.mul_scalar(&max_abs, rtol)?, atol)?;
    let normalized = client.div(&masked_error, &scale)?;
    let sq = client.mul(&normalized, &normalized)?;
    let sum_sq = client.sum(&sq, &[0], false)?;
    let sum_val: Vec<f64> = sum_sq.to_vec();

    Ok((sum_val[0] / n_diff).sqrt())
}

/// Compute step size adjustment factor.
pub(super) fn compute_step_factor(error: f64, order: usize) -> f64 {
    if error == 0.0 {
        return MAX_FACTOR;
    }
    let factor = SAFETY * error.powf(-1.0 / (order as f64 + 1.0));
    factor.clamp(MIN_FACTOR, MAX_FACTOR)
}

/// Adjust BDF order based on error.
pub(super) fn adjust_order(
    current_order: usize,
    error: f64,
    max_order: usize,
    history_len: usize,
) -> usize {
    let max_possible = (history_len - 1).min(max_order);
    if current_order >= max_possible {
        return current_order;
    }
    if error < 0.01 && current_order < max_possible {
        current_order + 1
    } else if error > 0.5 && current_order > 1 {
        current_order - 1
    } else {
        current_order
    }
}

/// Build the DAE result tensor.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_dae_result<R, C>(
    client: &C,
    t_values: &[f64],
    y_values: &[Tensor<R>],
    yp_values: &[Tensor<R>],
    success: bool,
    message: Option<String>,
    nfev: usize,
    njac: usize,
    n_ic_iter: usize,
    naccept: usize,
    nreject: usize,
    return_yp: bool,
) -> IntegrateResult<DAEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n_steps = t_values.len();
    let t_tensor = Tensor::<R>::from_slice(t_values, &[n_steps], client.device());

    let y_refs: Vec<&Tensor<R>> = y_values.iter().collect();
    let y_tensor = client
        .stack(&y_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack y tensors: {}", e),
        })?;

    let yp_tensor = if return_yp && !yp_values.is_empty() {
        let yp_refs: Vec<&Tensor<R>> = yp_values.iter().collect();
        Some(
            client
                .stack(&yp_refs, 0)
                .map_err(|e| IntegrateError::InvalidInput {
                    context: format!("Failed to stack yp tensors: {}", e),
                })?,
        )
    } else {
        None
    };

    Ok(DAEResultTensor {
        t: t_tensor,
        y: y_tensor,
        yp: yp_tensor,
        success,
        message,
        nfev,
        njac,
        n_ic_iter,
        naccept,
        nreject,
    })
}

pub(super) fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}
