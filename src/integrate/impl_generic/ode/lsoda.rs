//! LSODA - Automatic stiff/non-stiff method switching.
//!
//! Switches between Adams-Moulton (non-stiff) and BDF (stiff) based on
//! detected stiffness. Uses autograd for exact Jacobian computation in stiff mode.
//! All computation stays on device.
//!
//! # Unique Capability
//!
//! This is the only Rust LSODA implementation with automatic Jacobian computation.
//! Users write their ODE function using `DualTensor` operations, and the
//! solver computes exact Jacobians via forward-mode automatic differentiation.

use numr::autograd::DualTensor;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{LinalgOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::impl_generic::ode::{
    ODEResultParams, ODEResultTensor, build_ode_result, compute_error, compute_step_factor,
};
use crate::integrate::ode::{LSODAOptions, ODEMethod, ODEOptions};

use super::jacobian::{compute_jacobian_autograd, compute_norm_scalar, eval_primal};

// Adams-Moulton coefficients (predictor-corrector)
// AM-k: y_n = y_{n-1} + h * sum_{j=0}^{k} b_j * f_{n-j}
const AM_BETA: [[f64; 5]; 4] = [
    // AM-1 (Euler): y_n = y_{n-1} + h*f_n
    [1.0, 0.0, 0.0, 0.0, 0.0],
    // AM-2: y_n = y_{n-1} + h*(f_n + f_{n-1})/2
    [0.5, 0.5, 0.0, 0.0, 0.0],
    // AM-3
    [5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0, 0.0, 0.0],
    // AM-4
    [9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0, 0.0],
];

// Adams-Bashforth coefficients (predictor)
const AB_BETA: [[f64; 4]; 4] = [
    // AB-1 (Euler): y_n = y_{n-1} + h*f_{n-1}
    [1.0, 0.0, 0.0, 0.0],
    // AB-2
    [1.5, -0.5, 0.0, 0.0],
    // AB-3
    [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0, 0.0],
    // AB-4
    [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
];

// BDF coefficients (same as bdf.rs)
const BDF_ALPHA: [[f64; 6]; 5] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0, 0.0, 0.0],
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0, 0.0],
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
];

const BDF_BETA: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];

// Step size control
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

/// Method state for LSODA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)] // BDF is a standard acronym in ODE literature
enum MethodState {
    Adams, // Non-stiff
    BDF,   // Stiff
}

/// LSODA implementation with automatic method switching and autograd-based Jacobians.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - ODE right-hand side using `DualTensor` operations for automatic differentiation
/// * `t_span` - Integration interval [t_start, t_end]
/// * `y0` - Initial condition
/// * `options` - General ODE solver options
/// * `lsoda_options` - LSODA-specific options (switching thresholds)
pub fn lsoda_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
    lsoda_options: &LSODAOptions,
) -> IntegrateResult<ODEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let [t_start, t_end] = t_span;
    let device = client.device();
    let n = y0.shape()[0];

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize
    let mut t_val = t_start;
    let mut t = Tensor::<R>::from_slice(&[t_val], &[1], device);
    let mut y = y0.clone();

    // History buffers
    let mut y_history: Vec<Tensor<R>> = vec![y.clone()];
    let mut f_history: Vec<Tensor<R>> = Vec::new();

    // Compute initial f using primal evaluation
    let f0 = eval_primal(client, &f, &t, &y).map_err(to_integrate_err)?;
    f_history.push(f0.clone());

    // Initial step size
    let mut h = options.h0.unwrap_or_else(|| {
        let f_norm: f64 = compute_norm_scalar(client, &f0, 2.0).unwrap_or(1.0);
        let y_norm: f64 = compute_norm_scalar(client, &y, 2.0).unwrap_or(1.0);
        0.01 * (y_norm / f_norm.max(1e-10)).min(max_step)
    });
    h = h.clamp(min_step, max_step);

    // Method state
    let mut method_state = MethodState::Adams;
    let mut order = 1;
    let mut consecutive_rejects = 0;
    let mut consecutive_accepts = 0;

    // Jacobian cache (for BDF)
    let mut jacobian: Option<Tensor<R>> = None;

    // Results
    let mut t_values = vec![t_val];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    // Main loop
    while t_val < t_end {
        if naccept + nreject >= options.max_steps {
            return build_ode_result(
                client,
                ODEResultParams {
                    t_values: &t_values,
                    y_values: &y_values,
                    success: false,
                    message: Some(format!(
                        "Maximum steps ({}) exceeded at t = {:.6}",
                        options.max_steps, t_val
                    )),
                    nfev,
                    naccept,
                    nreject,
                },
                ODEMethod::LSODA,
            );
        }

        h = h.min(t_end - t_val);

        // Take a step based on current method
        let step_result = match method_state {
            MethodState::Adams => adams_step(
                client, &f, t_val, &y, &y_history, &f_history, h, order, options,
            ),
            MethodState::BDF => {
                // Ensure Jacobian is available using autograd (exact, no finite differences!)
                if jacobian.is_none() {
                    jacobian =
                        Some(compute_jacobian_autograd(client, &f, &t, &y).map_err(|e| {
                            IntegrateError::InvalidInput {
                                context: format!("Jacobian computation failed: {}", e),
                            }
                        })?);
                    nfev += n; // Forward-mode AD does n evaluations for n×n Jacobian
                }
                bdf_step(
                    client,
                    &f,
                    t_val,
                    &y,
                    &y_history,
                    h,
                    order.min(5),
                    jacobian.as_ref().unwrap(),
                    options,
                )
            }
        };

        let (y_new, f_new, error_val, step_nfev) = match step_result {
            Ok(r) => r,
            Err(_) => {
                // Step failed - reject and try smaller step
                h *= 0.5;
                nreject += 1;
                consecutive_rejects += 1;
                consecutive_accepts = 0;
                jacobian = None;

                if h < min_step {
                    return Err(IntegrateError::StepSizeTooSmall {
                        step: h,
                        t: t_val,
                        context: "LSODA step failed".to_string(),
                    });
                }
                continue;
            }
        };
        nfev += step_nfev;

        // Accept/reject
        if error_val <= 1.0 {
            // Accept
            t_val += h;
            t = Tensor::<R>::from_slice(&[t_val], &[1], device);
            y = y_new;

            // Update history
            y_history.insert(0, y.clone());
            f_history.insert(0, f_new);

            let max_history = match method_state {
                MethodState::Adams => lsoda_options.max_adams_order + 1,
                MethodState::BDF => lsoda_options.max_bdf_order + 1,
            };
            if y_history.len() > max_history {
                y_history.truncate(max_history);
                f_history.truncate(max_history);
            }

            t_values.push(t_val);
            y_values.push(y.clone());
            naccept += 1;
            consecutive_accepts += 1;
            consecutive_rejects = 0;

            // Order control
            let max_order = match method_state {
                MethodState::Adams => lsoda_options.max_adams_order.min(4),
                MethodState::BDF => lsoda_options.max_bdf_order.min(5),
            };
            order = adjust_order(order, error_val, max_order, y_history.len());
        } else {
            nreject += 1;
            consecutive_rejects += 1;
            consecutive_accepts = 0;
            jacobian = None;
        }

        // Stiffness detection and method switching
        if method_state == MethodState::Adams
            && consecutive_rejects >= lsoda_options.stiff_threshold
        {
            // Switch to BDF (stiff mode)
            method_state = MethodState::BDF;
            order = 1; // Reset order for new method
            consecutive_rejects = 0;
            jacobian = None;
        } else if method_state == MethodState::BDF
            && consecutive_accepts >= lsoda_options.nonstiff_threshold
        {
            // Switch back to Adams (non-stiff mode)
            method_state = MethodState::Adams;
            order = 1;
            consecutive_accepts = 0;
        }

        // Step size control
        let error_tensor = Tensor::<R>::from_slice(&[error_val], &[1], device);
        let factor =
            compute_step_factor(client, &error_tensor, order, SAFETY, MIN_FACTOR, MAX_FACTOR)
                .map_err(to_integrate_err)?;
        let factor_val: f64 = factor.to_vec()[0];
        h = (h * factor_val).clamp(min_step, max_step);
    }

    build_ode_result(
        client,
        ODEResultParams {
            t_values: &t_values,
            y_values: &y_values,
            success: true,
            message: None,
            nfev,
            naccept,
            nreject,
        },
        ODEMethod::LSODA,
    )
}

/// Adams-Moulton predictor-corrector step.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn adams_step<R, C, F>(
    client: &C,
    f: &F,
    t: f64,
    y: &Tensor<R>,
    _y_history: &[Tensor<R>],
    f_history: &[Tensor<R>],
    h: f64,
    order: usize,
    options: &ODEOptions,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let order_idx = (order - 1).min(3);

    // Predictor (Adams-Bashforth)
    let mut y_pred = y.clone();
    let ab_coeffs = &AB_BETA[order_idx];
    for (j, coeff) in ab_coeffs.iter().enumerate() {
        if *coeff != 0.0 && j < f_history.len() {
            let term = client
                .mul_scalar(&f_history[j], h * coeff)
                .map_err(to_integrate_err)?;
            y_pred = client.add(&y_pred, &term).map_err(to_integrate_err)?;
        }
    }

    // Evaluate at predicted point using primal evaluation
    let t_new = Tensor::<R>::from_slice(&[t + h], &[1], device);
    let f_pred = eval_primal(client, f, &t_new, &y_pred).map_err(to_integrate_err)?;

    // Corrector (Adams-Moulton)
    let am_coeffs = &AM_BETA[order_idx];
    let mut y_corr = y.clone();

    // First term: b_0 * f_new (at predicted point)
    let term0 = client
        .mul_scalar(&f_pred, h * am_coeffs[0])
        .map_err(to_integrate_err)?;
    y_corr = client.add(&y_corr, &term0).map_err(to_integrate_err)?;

    // Remaining terms from history
    for (j, coeff) in am_coeffs.iter().enumerate().skip(1) {
        if *coeff != 0.0 && j - 1 < f_history.len() {
            let term = client
                .mul_scalar(&f_history[j - 1], h * coeff)
                .map_err(to_integrate_err)?;
            y_corr = client.add(&y_corr, &term).map_err(to_integrate_err)?;
        }
    }

    // Final function evaluation using primal evaluation
    let f_corr = eval_primal(client, f, &t_new, &y_corr).map_err(to_integrate_err)?;

    // Error estimate: difference between predictor and corrector
    let y_err = client.sub(&y_corr, &y_pred).map_err(to_integrate_err)?;
    let error_tensor = compute_error(client, &y_corr, &y_err, y, options.rtol, options.atol)
        .map_err(to_integrate_err)?;
    let error_val: f64 = error_tensor.to_vec()[0];

    Ok((y_corr, f_corr, error_val, 2)) // 2 function evaluations
}

/// BDF step with Newton iteration.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn bdf_step<R, C, F>(
    client: &C,
    f: &F,
    t: f64,
    y: &Tensor<R>,
    y_history: &[Tensor<R>],
    h: f64,
    order: usize,
    jacobian: &Tensor<R>,
    options: &ODEOptions,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let n = y.shape()[0];
    let order_idx = (order - 1).min(4);
    let alpha = &BDF_ALPHA[order_idx];
    let beta = BDF_BETA[order_idx];

    // Compute RHS from history
    // Note: y_{n-i} corresponds to y_history[i-1] since y_history[0] = y_{n-1}
    let mut rhs = client.mul_scalar(y, 0.0).map_err(to_integrate_err)?;
    for i in 1..=order {
        if i <= y_history.len() {
            let term = client
                .mul_scalar(&y_history[i - 1], -alpha[i])
                .map_err(to_integrate_err)?;
            rhs = client.add(&rhs, &term).map_err(to_integrate_err)?;
        }
    }

    // Build iteration matrix using numr's eye() for GPU-efficient identity construction
    let identity = client.eye(n, None, DType::F64).map_err(to_integrate_err)?;
    let scaled_identity = client
        .mul_scalar(&identity, alpha[0])
        .map_err(to_integrate_err)?;
    let scaled_j = client
        .mul_scalar(jacobian, h * beta)
        .map_err(to_integrate_err)?;
    let m_matrix = client
        .sub(&scaled_identity, &scaled_j)
        .map_err(to_integrate_err)?;

    // Newton iteration
    let t_new = Tensor::<R>::from_slice(&[t + h], &[1], device);
    let mut y_iter = y.clone();
    let mut nfev = 0;
    let max_iter = 10;

    for _ in 0..max_iter {
        // Function evaluation using primal evaluation
        let f_iter = eval_primal(client, f, &t_new, &y_iter).map_err(to_integrate_err)?;
        nfev += 1;

        let term1 = client
            .mul_scalar(&y_iter, alpha[0])
            .map_err(to_integrate_err)?;
        let term2 = client
            .mul_scalar(&f_iter, h * beta)
            .map_err(to_integrate_err)?;
        let residual = client
            .sub(&client.sub(&term1, &term2).map_err(to_integrate_err)?, &rhs)
            .map_err(to_integrate_err)?;

        let res_norm: f64 =
            compute_norm_scalar(client, &residual, 2.0).map_err(to_integrate_err)?;
        let y_norm: f64 = compute_norm_scalar(client, &y_iter, 2.0).map_err(to_integrate_err)?;

        if res_norm < 1e-6 * (1.0 + y_norm) {
            // Converged - compute error and return
            let y_err = client.sub(&y_iter, y).map_err(to_integrate_err)?;
            let y_err_scaled = client.mul_scalar(&y_err, 0.1).map_err(to_integrate_err)?;
            let error_tensor = compute_error(
                client,
                &y_iter,
                &y_err_scaled,
                y,
                options.rtol,
                options.atol,
            )
            .map_err(to_integrate_err)?;
            let error_val: f64 = error_tensor.to_vec()[0];

            // Final function evaluation using primal evaluation
            let f_final = eval_primal(client, f, &t_new, &y_iter).map_err(to_integrate_err)?;
            nfev += 1;

            return Ok((y_iter, f_final, error_val, nfev));
        }

        // Solve for correction
        let neg_res = client
            .mul_scalar(&residual, -1.0)
            .map_err(to_integrate_err)?;
        let neg_res_col = neg_res.reshape(&[n, 1]).map_err(to_integrate_err)?;
        let delta_col = client
            .solve(&m_matrix, &neg_res_col)
            .map_err(to_integrate_err)?;
        let delta = delta_col.reshape(&[n]).map_err(to_integrate_err)?;

        y_iter = client.add(&y_iter, &delta).map_err(to_integrate_err)?;
    }

    // Did not converge
    Err(IntegrateError::DidNotConverge {
        iterations: max_iter,
        tolerance: 1e-6,
        context: "BDF Newton iteration".to_string(),
    })
}

/// Adjust order based on error.
fn adjust_order(current: usize, error: f64, max_order: usize, history_len: usize) -> usize {
    let max_possible = (history_len - 1).min(max_order);

    if error < 0.01 && current < max_possible {
        current + 1
    } else if error > 0.5 && current > 1 {
        current - 1
    } else {
        current
    }
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::dual_mul_scalar;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_lsoda_nonstiff() {
        let (device, client) = setup();

        // Non-stiff: dy/dt = -y using dual operations
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = lsoda_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -1.0, c),
            [0.0, 5.0],
            &y0,
            &ODEOptions::with_tolerances(1e-4, 1e-6),
            &LSODAOptions::default(),
        )
        .unwrap();

        assert!(result.success);

        let y_final = result.y_final_vec();
        let exact = (-5.0_f64).exp();

        assert!(
            (y_final[0] - exact).abs() < 1e-3,
            "y_final = {}, exact = {}",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_lsoda_stiff() {
        let (device, client) = setup();

        // Stiff: dy/dt = -1000*y using dual operations
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = lsoda_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -1000.0, c),
            [0.0, 0.1],
            &y0,
            &ODEOptions::with_tolerances(1e-4, 1e-6),
            &LSODAOptions::default(),
        )
        .unwrap();

        assert!(
            result.success,
            "LSODA should handle stiff: {:?}",
            result.message
        );

        let y_final = result.y_final_vec();
        // exp(-1000 * 0.1) = exp(-100) ≈ 3.7e-44
        // With solver tolerances, just verify it's very small
        assert!(
            y_final[0].abs() < 1e-5,
            "y_final = {}, should be close to 0",
            y_final[0]
        );
    }
}
