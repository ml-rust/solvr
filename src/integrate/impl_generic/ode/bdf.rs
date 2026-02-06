//! Backward Differentiation Formula (BDF) solver for stiff ODEs.
//!
//! Implements variable-order BDF methods (orders 1-5) with Newton iteration
//! for solving the implicit equations. Uses autograd for exact Jacobians.
//!
//! # Unique Capability
//!
//! This is the only Rust ODE solver with automatic Jacobian computation.
//! Users write their ODE function using `DualTensor` operations, and the
//! solver computes exact Jacobians via forward-mode automatic differentiation.
//!
//! # Example
//!
//! ```ignore
//! use numr::autograd::DualTensor;
//! use numr::autograd::dual_ops::dual_mul_scalar;
//!
//! // Stiff decay: dy/dt = -1000*y
//! let f = |_t: &DualTensor<R>, y: &DualTensor<R>, c: &C| {
//!     dual_mul_scalar(y, -1000.0, c)
//! };
//!
//! let result = bdf_impl(&client, f, [0.0, 1.0], &y0, &opts, &bdf_opts)?;
//! ```

use numr::autograd::DualTensor;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::Runtime;
#[allow(unused_imports)]
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

#[cfg(feature = "sparse")]
use super::direct_solver::DirectSparseSolver;
#[cfg(feature = "sparse")]
use super::sparse_utils::{create_direct_solver, solve_sparse_system};

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::impl_generic::ode::{
    ODEResultParams, ODEResultTensor, build_ode_result, compute_error, compute_step_factor,
};
use crate::integrate::ode::{BDFOptions, ODEMethod, ODEOptions};

use super::jacobian::{compute_jacobian_autograd, compute_norm_scalar, eval_primal};
use super::stiff_client::StiffSolverClient;

// BDF coefficients for orders 1-5
const BDF_ALPHA: [[f64; 6]; 5] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0, 0.0, 0.0],
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0, 0.0],
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
];

const BDF_BETA: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];

const BDF_ERROR_COEFF: [f64; 5] = [1.0 / 2.0, 2.0 / 9.0, 3.0 / 22.0, 12.0 / 125.0, 10.0 / 137.0];

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

/// BDF solver implementation with automatic Jacobian computation.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - ODE right-hand side using `DualTensor` operations for automatic differentiation
/// * `t_span` - Integration interval [t_start, t_end]
/// * `y0` - Initial condition
/// * `options` - General ODE solver options
/// * `bdf_options` - BDF-specific options
///
/// # Function Signature
///
/// The ODE function `f(t, y, client)` must use `DualTensor` and `dual_*` operations
/// from `numr::autograd::dual_ops`. This enables automatic Jacobian computation
/// without finite differences.
pub fn bdf_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
    bdf_options: &BDFOptions<R>,
) -> IntegrateResult<ODEResultTensor<R>>
where
    R: Runtime,
    C: StiffSolverClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let [t_start, t_end] = t_span;
    let device = client.device();
    let n = y0.shape()[0];

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);
    let max_order = bdf_options.max_order.clamp(1, 5);

    // Initialize state
    let mut t_val = t_start;
    let t = Tensor::<R>::from_slice(&[t_val], &[1], device);
    let mut y = y0.clone();

    // History buffer for multistep method
    let mut y_history: Vec<Tensor<R>> = vec![y.clone()];
    let mut f_history: Vec<Tensor<R>> = Vec::new();

    // Compute initial f (using primal evaluation)
    let f0 = eval_primal(client, &f, &t, &y).map_err(|e| IntegrateError::InvalidInput {
        context: format!("RHS function error: {}", e),
    })?;
    f_history.push(f0.clone());

    // Initial step size
    let mut h = options.h0.unwrap_or_else(|| {
        let f_norm: f64 = compute_norm_scalar(client, &f0, 2.0).unwrap_or(1.0);
        let y_norm: f64 = compute_norm_scalar(client, &y, 2.0).unwrap_or(1.0);
        0.01 * (y_norm / f_norm.max(1e-10)).min(max_step)
    });
    h = h.clamp(min_step, max_step);

    let mut order = 1;

    // Results storage
    let mut t_values = vec![t_val];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    // Jacobian (recomputed periodically)
    let mut jacobian: Option<Tensor<R>> = None;
    let mut steps_since_jacobian = 0;
    let jacobian_update_interval = 5;

    // Direct sparse LU solver (created when strategy is DirectLU or Auto)
    #[cfg(feature = "sparse")]
    let mut direct_solver = create_direct_solver(&bdf_options.sparse_jacobian, n);

    // Main integration loop
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
                ODEMethod::BDF,
            );
        }

        h = h.min(t_end - t_val);

        // Compute predictor
        let y_pred = compute_predictor(client, &y_history, &f_history, order, h)?;

        // Update Jacobian using autograd (exact, no finite differences!)
        if jacobian.is_none() || steps_since_jacobian >= jacobian_update_interval {
            let t_new = Tensor::<R>::from_slice(&[t_val + h], &[1], device);
            jacobian = Some(
                compute_jacobian_autograd(client, &f, &t_new, &y_pred).map_err(|e| {
                    IntegrateError::InvalidInput {
                        context: format!("Jacobian computation failed: {}", e),
                    }
                })?,
            );
            steps_since_jacobian = 0;
            nfev += n; // Forward-mode AD does n evaluations for n×n Jacobian
        }

        // Newton iteration
        let (y_new, converged, newton_iters) = newton_iteration(
            client,
            &f,
            t_val + h,
            &y_pred,
            &y_history,
            order,
            h,
            jacobian.as_ref().unwrap(),
            bdf_options,
            #[cfg(feature = "sparse")]
            &mut direct_solver,
        )?;
        nfev += newton_iters;

        if !converged {
            h *= 0.5;
            nreject += 1;
            jacobian = None;

            if h < min_step {
                return Err(IntegrateError::StepSizeTooSmall {
                    step: h,
                    t: t_val,
                    context: "BDF Newton iteration failed to converge".to_string(),
                });
            }
            continue;
        }

        // Compute error estimate
        let t_new = Tensor::<R>::from_slice(&[t_val + h], &[1], device);
        let f_new = eval_primal(client, &f, &t_new, &y_new).map_err(to_integrate_err)?;
        nfev += 1;

        let error_tensor = estimate_error(client, &y_new, &y_pred, order)?;
        let error = compute_error(
            client,
            &y_new,
            &error_tensor,
            &y,
            options.rtol,
            options.atol,
        )
        .map_err(to_integrate_err)?;
        let error_val: f64 = error.item().map_err(to_integrate_err)?;

        if error_val <= 1.0 {
            // Accept step
            t_val += h;
            y = y_new;

            y_history.insert(0, y.clone());
            f_history.insert(0, f_new);

            let history_len = max_order + 1;
            if y_history.len() > history_len {
                y_history.truncate(history_len);
                f_history.truncate(history_len);
            }

            t_values.push(t_val);
            y_values.push(y.clone());
            naccept += 1;
            steps_since_jacobian += 1;

            order = adjust_order(order, error_val, max_order, y_history.len());
        } else {
            nreject += 1;
            jacobian = None;
        }

        // Step size control
        let factor = compute_step_factor(client, &error, order, SAFETY, MIN_FACTOR, MAX_FACTOR)
            .map_err(to_integrate_err)?;
        let factor_val: f64 = factor.item().map_err(to_integrate_err)?;
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
        ODEMethod::BDF,
    )
}

/// Compute predictor from history.
fn compute_predictor<R, C>(
    client: &C,
    y_history: &[Tensor<R>],
    f_history: &[Tensor<R>],
    order: usize,
    h: f64,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    if y_history.is_empty() {
        return Err(IntegrateError::InvalidInput {
            context: "Empty history".to_string(),
        });
    }

    let y_n = &y_history[0];

    if f_history.is_empty() {
        return Ok(y_n.clone());
    }

    let f_n = &f_history[0];

    if order == 1 || y_history.len() < 2 {
        let h_f = client.mul_scalar(f_n, h).map_err(to_integrate_err)?;
        client.add(y_n, &h_f).map_err(to_integrate_err)
    } else {
        let f_nm1 = &f_history[1.min(f_history.len() - 1)];
        let term1 = client.mul_scalar(f_n, 1.5 * h).map_err(to_integrate_err)?;
        let term2 = client
            .mul_scalar(f_nm1, -0.5 * h)
            .map_err(to_integrate_err)?;
        let df = client.add(&term1, &term2).map_err(to_integrate_err)?;
        client.add(y_n, &df).map_err(to_integrate_err)
    }
}

/// Newton iteration for the BDF implicit equation.
#[allow(clippy::too_many_arguments)]
fn newton_iteration<R, C, F>(
    client: &C,
    f: &F,
    t_new: f64,
    y_pred: &Tensor<R>,
    y_history: &[Tensor<R>],
    order: usize,
    h: f64,
    jacobian: &Tensor<R>,
    options: &BDFOptions<R>,
    #[cfg(feature = "sparse")] direct_solver: &mut Option<DirectSparseSolver<R>>,
) -> IntegrateResult<(Tensor<R>, bool, usize)>
where
    R: Runtime,
    C: StiffSolverClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let order_idx = (order - 1).min(4);
    let alpha = &BDF_ALPHA[order_idx];
    let beta = BDF_BETA[order_idx];

    // Compute RHS: sum_{i=1}^{order} (-alpha[i]) * y_{n-i}
    let mut rhs = client
        .mul_scalar(&y_history[0], 0.0)
        .map_err(to_integrate_err)?;
    for i in 1..=order {
        if i <= y_history.len() {
            let term = client
                .mul_scalar(&y_history[i - 1], -alpha[i])
                .map_err(to_integrate_err)?;
            rhs = client.add(&rhs, &term).map_err(to_integrate_err)?;
        }
    }

    // Iteration matrix: M = alpha[0]*I - h*beta*J
    let n = y_pred.shape()[0];
    let identity = client.eye(n, None, DType::F64).map_err(to_integrate_err)?;
    let scaled_identity = client
        .mul_scalar(&identity, alpha[0])
        .map_err(to_integrate_err)?;
    let scaled_j = client
        .mul_scalar(jacobian, h * beta)
        .map_err(to_integrate_err)?;
    let iteration_matrix = client
        .sub(&scaled_identity, &scaled_j)
        .map_err(to_integrate_err)?;

    let mut y_iter = y_pred.clone();
    let t_tensor = Tensor::<R>::from_slice(&[t_new], &[1], device);
    let mut nfev = 0;

    for _ in 0..options.max_newton_iter {
        let f_iter = eval_primal(client, f, &t_tensor, &y_iter).map_err(to_integrate_err)?;
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

        if res_norm < options.newton_tol * (1.0 + y_norm) {
            return Ok((y_iter, true, nfev));
        }

        let neg_residual = client
            .mul_scalar(&residual, -1.0)
            .map_err(to_integrate_err)?;
        let neg_res_col = neg_residual.reshape(&[n, 1]).map_err(to_integrate_err)?;
        let delta_col = solve_bdf_linear(
            client,
            &iteration_matrix,
            &neg_res_col,
            &options.sparse_jacobian,
            #[cfg(feature = "sparse")]
            direct_solver,
        )
        .map_err(to_integrate_err)?;
        let delta = delta_col.reshape(&[n]).map_err(to_integrate_err)?;

        y_iter = client.add(&y_iter, &delta).map_err(to_integrate_err)?;
    }

    Ok((y_iter, false, nfev))
}

fn estimate_error<R, C>(
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

fn adjust_order(current_order: usize, error: f64, max_order: usize, history_len: usize) -> usize {
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

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

// With sparse feature: support dense, GMRES, and direct LU solvers
#[cfg(feature = "sparse")]
fn solve_bdf_linear<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    sparse_config: &crate::integrate::ode::SparseJacobianConfig<R>,
    direct_solver: &mut Option<DirectSparseSolver<R>>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: StiffSolverClient<R>,
{
    if !sparse_config.enabled {
        return client.solve(m_dense, b);
    }

    solve_sparse_system(
        client,
        m_dense,
        b,
        sparse_config,
        direct_solver,
        None,
        "BDF",
    )
}

// Without sparse feature: dense-only solver
#[cfg(not(feature = "sparse"))]
fn solve_bdf_linear<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    _sparse_config: &crate::integrate::ode::SparseJacobianConfig<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: numr::ops::LinalgOps<R> + RuntimeClient<R>,
{
    client.solve(m_dense, b)
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
    fn test_bdf_exponential_decay() {
        let (_device, client) = setup();

        // dy/dt = -y, y(0) = 1, solution: y(t) = exp(-t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], client.device());

        let result = bdf_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -1.0, c),
            [0.0, 5.0],
            &y0,
            &ODEOptions::with_tolerances(1e-4, 1e-6),
            &BDFOptions::<CpuRuntime>::default(),
        )
        .unwrap();

        assert!(result.success, "BDF should succeed: {:?}", result.message);

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
    fn test_bdf_stiff_decay() {
        let (_device, client) = setup();

        // Stiff system: dy/dt = -1000*y
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], client.device());

        let result = bdf_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -1000.0, c),
            [0.0, 0.1],
            &y0,
            &ODEOptions::with_tolerances(1e-4, 1e-6),
            &BDFOptions::<CpuRuntime>::default(),
        )
        .unwrap();

        assert!(
            result.success,
            "BDF should handle stiff system: {:?}",
            result.message
        );

        let y_final = result.y_final_vec();
        assert!(
            y_final[0].abs() < 1e-6,
            "y_final = {}, should be close to 0",
            y_final[0]
        );
    }

    #[test]
    fn test_bdf_system() {
        let (_device, client) = setup();

        // System: dy1/dt = -y1, dy2/dt = y1 - y2
        // Using dual operations for coupled system
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], client.device());

        let mut opts = ODEOptions::with_tolerances(1e-4, 1e-6);
        opts.max_steps = 50000;

        let result = bdf_impl(
            &client,
            |_t, y, c| {
                // For coupled systems, extract primal for indexing, then build dual result
                let y_data: Vec<f64> = y.primal().to_vec();
                let dy = [-y_data[0], y_data[0] - y_data[1]];
                // Create output DualTensor - tangent will be computed by autograd
                let dy_tensor = Tensor::<CpuRuntime>::from_slice(&dy, &[2], c.device());
                Ok(DualTensor::new(dy_tensor, None))
            },
            [0.0, 2.0],
            &y0,
            &opts,
            &BDFOptions::<CpuRuntime>::default(),
        )
        .unwrap();

        assert!(result.success, "BDF system failed: {:?}", result.message);

        let y_final = result.y_final_vec();
        let t_final: f64 = 2.0;
        let y1_exact = (-t_final).exp();
        let y2_exact = t_final * (-t_final).exp();

        assert!(
            (y_final[0] - y1_exact).abs() < 5e-3,
            "y1_final = {}, exact = {}",
            y_final[0],
            y1_exact
        );
        assert!(
            (y_final[1] - y2_exact).abs() < 5e-3,
            "y2_final = {}, exact = {}",
            y_final[1],
            y2_exact
        );
    }
}
