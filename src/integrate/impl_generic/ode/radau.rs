//! Radau IIA order 5 implicit Runge-Kutta solver.
//!
//! A 3-stage implicit method particularly effective for very stiff problems.
//! Uses autograd for exact Jacobian computation. All computation stays on device.
//!
//! # Unique Capability
//!
//! This is the only Rust Radau solver with automatic Jacobian computation.
//! Users write their ODE function using `DualTensor` operations, and the
//! solver computes exact Jacobians via forward-mode automatic differentiation.

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
use crate::integrate::ode::{ODEMethod, ODEOptions, RadauOptions};

use super::jacobian::{compute_jacobian_autograd, compute_norm_scalar, eval_primal};
use super::stiff_client::StiffSolverClient;

// Radau IIA order 5 coefficients (3-stage)
// sqrt(6) ≈ 2.449489742783178
const SQRT6: f64 = 2.449489742783178;

// Butcher tableau nodes (c values)
// C1 = (4 - sqrt(6)) / 10 ≈ 0.1550510257216822
// C2 = (4 + sqrt(6)) / 10 ≈ 0.6449489742783178
const C1: f64 = (4.0 - SQRT6) / 10.0;
const C2: f64 = (4.0 + SQRT6) / 10.0;
const C3: f64 = 1.0;

// Butcher tableau A matrix (implicit)
// A[i,j] coefficient for stage j in stage i equation
const A11: f64 = (88.0 - 7.0 * SQRT6) / 360.0;
const A12: f64 = (296.0 - 169.0 * SQRT6) / 1800.0;
const A13: f64 = (-2.0 + 3.0 * SQRT6) / 225.0;

const A21: f64 = (296.0 + 169.0 * SQRT6) / 1800.0;
const A22: f64 = (88.0 + 7.0 * SQRT6) / 360.0;
const A23: f64 = (-2.0 - 3.0 * SQRT6) / 225.0;

const A31: f64 = (16.0 - SQRT6) / 36.0;
const A32: f64 = (16.0 + SQRT6) / 36.0;
const A33: f64 = 1.0 / 9.0;

// Weights (b values) - same as last row of A for stiffly accurate method
const B1: f64 = (16.0 - SQRT6) / 36.0;
const B2: f64 = (16.0 + SQRT6) / 36.0;
const B3: f64 = A33;

// Error estimator weights (embedded lower-order method)
const E1: f64 = -13.0 - 7.0 * SQRT6;
const E2: f64 = -13.0 + 7.0 * SQRT6;
const E3: f64 = -1.0;
const E_DENOM: f64 = 3.0;

// Step size control
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

/// Radau IIA order 5 implementation with automatic Jacobian computation.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - ODE right-hand side using `DualTensor` operations for automatic differentiation
/// * `t_span` - Integration interval [t_start, t_end]
/// * `y0` - Initial condition
/// * `options` - General ODE solver options
/// * `radau_options` - Radau-specific options
pub fn radau_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
    radau_options: &RadauOptions<R>,
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

    // Initialize
    let mut t_val = t_start;
    let mut t = Tensor::<R>::from_slice(&[t_val], &[1], device);
    let mut y = y0.clone();

    // Compute initial f using primal evaluation
    let f0 = eval_primal(client, &f, &t, &y).map_err(to_integrate_err)?;

    // Initial step size
    let mut h = options.h0.unwrap_or_else(|| {
        let f_norm: f64 = compute_norm_scalar(client, &f0, 2.0).unwrap_or(1.0);
        let y_norm: f64 = compute_norm_scalar(client, &y, 2.0).unwrap_or(1.0);
        0.01 * (y_norm / f_norm.max(1e-10)).min(max_step)
    });
    h = h.clamp(min_step, max_step);

    // Results storage
    let mut t_values = vec![t_val];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    // Jacobian cache
    let mut jacobian: Option<Tensor<R>> = None;
    let mut steps_since_jacobian = 0;
    let jacobian_update_interval = if radau_options.simplified_newton {
        10
    } else {
        1
    };

    // Direct sparse LU solver (created when strategy is DirectLU or Auto)
    #[cfg(feature = "sparse")]
    let mut direct_solver = create_direct_solver(&radau_options.sparse_jacobian, n);

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
                ODEMethod::Radau,
            );
        }

        // Adjust step for end point
        h = h.min(t_end - t_val);

        // Update Jacobian using autograd (exact, no finite differences!)
        if jacobian.is_none() || steps_since_jacobian >= jacobian_update_interval {
            jacobian = Some(compute_jacobian_autograd(client, &f, &t, &y).map_err(|e| {
                IntegrateError::InvalidInput {
                    context: format!("Jacobian computation failed: {}", e),
                }
            })?);
            steps_since_jacobian = 0;
            nfev += n; // Forward-mode AD does n evaluations for n×n Jacobian
        }

        // Solve implicit system for stages
        let (k1, k2, k3, converged, newton_iters) = solve_radau_stages(
            client,
            &f,
            t_val,
            &y,
            h,
            jacobian.as_ref().unwrap(),
            radau_options,
            #[cfg(feature = "sparse")]
            &mut direct_solver,
        )?;
        nfev += newton_iters;

        if !converged {
            // Newton failed - reduce step
            h *= 0.5;
            nreject += 1;
            jacobian = None;

            if h < min_step {
                return Err(IntegrateError::StepSizeTooSmall {
                    step: h,
                    t: t_val,
                    context: "Radau Newton iteration failed".to_string(),
                });
            }
            continue;
        }

        // Compute new solution: y_new = y + h*(b1*k1 + b2*k2 + b3*k3)
        let term1 = client.mul_scalar(&k1, h * B1).map_err(to_integrate_err)?;
        let term2 = client.mul_scalar(&k2, h * B2).map_err(to_integrate_err)?;
        let term3 = client.mul_scalar(&k3, h * B3).map_err(to_integrate_err)?;
        let increment = client
            .add(
                &client.add(&term1, &term2).map_err(to_integrate_err)?,
                &term3,
            )
            .map_err(to_integrate_err)?;
        let y_new = client.add(&y, &increment).map_err(to_integrate_err)?;

        // Error estimate
        let y_err = compute_radau_error(client, &k1, &k2, &k3, h)?;
        let error = compute_error(client, &y_new, &y_err, &y, options.rtol, options.atol)
            .map_err(to_integrate_err)?;
        let error_val: f64 = error.item().map_err(to_integrate_err)?;

        // Accept/reject
        if error_val <= 1.0 {
            t_val += h;
            t = Tensor::<R>::from_slice(&[t_val], &[1], device);
            y = y_new;

            t_values.push(t_val);
            y_values.push(y.clone());
            naccept += 1;
            steps_since_jacobian += 1;
        } else {
            nreject += 1;
            if !radau_options.simplified_newton {
                jacobian = None;
            }
        }

        // Step size control
        let factor = compute_step_factor(client, &error, 5, SAFETY, MIN_FACTOR, MAX_FACTOR)
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
        ODEMethod::Radau,
    )
}

/// Solve the Radau implicit system for all three stages.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn solve_radau_stages<R, C, F>(
    client: &C,
    f: &F,
    t: f64,
    y: &Tensor<R>,
    h: f64,
    jacobian: &Tensor<R>,
    options: &RadauOptions<R>,
    #[cfg(feature = "sparse")] direct_solver: &mut Option<DirectSparseSolver<R>>,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, Tensor<R>, bool, usize)>
where
    R: Runtime,
    C: StiffSolverClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let n = y.shape()[0];

    // Stage times
    let t1 = t + C1 * h;
    let t2 = t + C2 * h;
    let t3 = t + C3 * h;

    let t1_tensor = Tensor::<R>::from_slice(&[t1], &[1], device);
    let t2_tensor = Tensor::<R>::from_slice(&[t2], &[1], device);
    let t3_tensor = Tensor::<R>::from_slice(&[t3], &[1], device);

    // Initial guess for stages (explicit Euler) using primal evaluation
    let t_tensor = Tensor::<R>::from_slice(&[t], &[1], device);
    let f_y = eval_primal(client, f, &t_tensor, y).map_err(to_integrate_err)?;
    let mut k1 = f_y.clone();
    let mut k2 = f_y.clone();
    let mut k3 = f_y;

    // Build the iteration matrix for simplified Newton
    // Use numr's eye() for GPU-efficient identity matrix construction
    let gamma = A33; // Use diagonal element
    let scaled_j = client
        .mul_scalar(jacobian, h * gamma)
        .map_err(to_integrate_err)?;
    let identity = client.eye(n, None, DType::F64).map_err(to_integrate_err)?;
    let m_matrix = client.sub(&identity, &scaled_j).map_err(to_integrate_err)?;

    let mut nfev = 0;

    // Newton iteration
    for _ in 0..options.max_newton_iter {
        // Stage values
        // Y_i = y + h * sum_j(a_ij * k_j)
        let y1 = compute_stage_value(client, y, h, &k1, &k2, &k3, A11, A12, A13)?;
        let y2 = compute_stage_value(client, y, h, &k1, &k2, &k3, A21, A22, A23)?;
        let y3 = compute_stage_value(client, y, h, &k1, &k2, &k3, A31, A32, A33)?;

        // Function evaluations using primal evaluation
        let f1 = eval_primal(client, f, &t1_tensor, &y1).map_err(to_integrate_err)?;
        let f2 = eval_primal(client, f, &t2_tensor, &y2).map_err(to_integrate_err)?;
        let f3 = eval_primal(client, f, &t3_tensor, &y3).map_err(to_integrate_err)?;
        nfev += 3;

        // Residuals: r_i = k_i - f_i
        let r1 = client.sub(&k1, &f1).map_err(to_integrate_err)?;
        let r2 = client.sub(&k2, &f2).map_err(to_integrate_err)?;
        let r3 = client.sub(&k3, &f3).map_err(to_integrate_err)?;

        // Check convergence using norm computation
        let r1_norm: f64 = compute_norm_scalar(client, &r1, 2.0).map_err(to_integrate_err)?;
        let r2_norm: f64 = compute_norm_scalar(client, &r2, 2.0).map_err(to_integrate_err)?;
        let r3_norm: f64 = compute_norm_scalar(client, &r3, 2.0).map_err(to_integrate_err)?;
        let max_res = r1_norm.max(r2_norm).max(r3_norm);

        let k1_norm: f64 = compute_norm_scalar(client, &k1, 2.0).map_err(to_integrate_err)?;
        let k2_norm: f64 = compute_norm_scalar(client, &k2, 2.0).map_err(to_integrate_err)?;
        let k3_norm: f64 = compute_norm_scalar(client, &k3, 2.0).map_err(to_integrate_err)?;
        let k_norm = (k1_norm + k2_norm + k3_norm) / 3.0;

        if max_res < options.newton_tol * (1.0 + k_norm) {
            return Ok((k1, k2, k3, true, nfev));
        }

        // Solve for corrections using simplified Newton
        let neg_r1 = client.mul_scalar(&r1, -1.0).map_err(to_integrate_err)?;
        let neg_r2 = client.mul_scalar(&r2, -1.0).map_err(to_integrate_err)?;
        let neg_r3 = client.mul_scalar(&r3, -1.0).map_err(to_integrate_err)?;

        let dk1 = solve_linear(
            client,
            &m_matrix,
            &neg_r1,
            &options.sparse_jacobian,
            #[cfg(feature = "sparse")]
            direct_solver,
        )?;
        let dk2 = solve_linear(
            client,
            &m_matrix,
            &neg_r2,
            &options.sparse_jacobian,
            #[cfg(feature = "sparse")]
            direct_solver,
        )?;
        let dk3 = solve_linear(
            client,
            &m_matrix,
            &neg_r3,
            &options.sparse_jacobian,
            #[cfg(feature = "sparse")]
            direct_solver,
        )?;

        // Update stages
        k1 = client.add(&k1, &dk1).map_err(to_integrate_err)?;
        k2 = client.add(&k2, &dk2).map_err(to_integrate_err)?;
        k3 = client.add(&k3, &dk3).map_err(to_integrate_err)?;
    }

    Ok((k1, k2, k3, false, nfev))
}

/// Compute stage value Y_i = y + h*(a1*k1 + a2*k2 + a3*k3).
#[allow(clippy::too_many_arguments)]
fn compute_stage_value<R, C>(
    client: &C,
    y: &Tensor<R>,
    h: f64,
    k1: &Tensor<R>,
    k2: &Tensor<R>,
    k3: &Tensor<R>,
    a1: f64,
    a2: f64,
    a3: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let term1 = client.mul_scalar(k1, h * a1)?;
    let term2 = client.mul_scalar(k2, h * a2)?;
    let term3 = client.mul_scalar(k3, h * a3)?;

    let sum12 = client.add(&term1, &term2)?;
    let sum123 = client.add(&sum12, &term3)?;
    client.add(y, &sum123)
}

/// Solve linear system M*x = b using dense or sparse solver.
///
/// # Arguments
///
/// * `client` - Runtime client with linear algebra operations
/// * `m_dense` - Dense iteration matrix (I - h*γ*J)
/// * `b` - Right-hand side vector
/// * `sparse_config` - Optional sparse Jacobian configuration
///
/// # Returns
///
/// Solution vector x such that M*x = b
///
/// # Method
///
/// - If `sparse_config.enabled == false`: Uses dense LU factorization via `client.solve()`
/// - If `sparse_config.enabled == true`: Converts dense M to CSR format and uses GMRES
///   with optional ILU(0) preconditioning
// With sparse feature: support dense, GMRES, and direct LU solvers
#[cfg(feature = "sparse")]
fn solve_linear<R, C>(
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
        // Dense path - Radau requires reshape
        let n = b.shape()[0];
        let b_col = b.reshape(&[n, 1])?;
        let x_col = client.solve(m_dense, &b_col)?;
        return x_col.reshape(&[n]);
    }

    // Sparse path - use pattern if available for Radau
    solve_sparse_system(
        client,
        m_dense,
        b,
        sparse_config,
        direct_solver,
        sparse_config.pattern.as_ref(),
        "Radau",
    )
}

// Without sparse feature: dense-only solver
#[cfg(not(feature = "sparse"))]
fn solve_linear<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    _sparse_config: &crate::integrate::ode::SparseJacobianConfig<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: numr::ops::LinalgOps<R> + RuntimeClient<R>,
{
    let n = b.shape()[0];
    let b_col = b.reshape(&[n, 1])?;
    let x_col = client.solve(m_dense, &b_col)?;
    x_col.reshape(&[n])
}

/// Compute Radau error estimate.
fn compute_radau_error<R, C>(
    client: &C,
    k1: &Tensor<R>,
    k2: &Tensor<R>,
    k3: &Tensor<R>,
    h: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    // Error estimate: h * (e1*k1 + e2*k2 + e3*k3) / denom
    let coeff = h / E_DENOM;
    let term1 = client.mul_scalar(k1, coeff * E1)?;
    let term2 = client.mul_scalar(k2, coeff * E2)?;
    let term3 = client.mul_scalar(k3, coeff * E3)?;

    let sum12 = client.add(&term1, &term2)?;
    client.add(&sum12, &term3)
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
    fn test_radau_exponential() {
        let (device, client) = setup();

        // Radau is optimized for very stiff problems
        // Use same parameters that work in very_stiff test
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let mut opts = ODEOptions::with_tolerances(1e-2, 1e-4);
        opts.max_steps = 50000;

        // Stiff problem: dy/dt = -1000*y using dual operations
        let result = radau_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -1000.0, c),
            [0.0, 0.01],
            &y0,
            &opts,
            &RadauOptions::<CpuRuntime>::default(),
        )
        .unwrap();

        assert!(result.success, "Radau should succeed: {:?}", result.message);

        let y_final = result.y_final_vec();
        let exact = (-10.0_f64).exp(); // exp(-1000 * 0.01)

        assert!(
            (y_final[0] - exact).abs() < 1e-3,
            "y_final = {}, exact = {}",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_radau_very_stiff() {
        let (device, client) = setup();

        // Very stiff: dy/dt = -10000*y
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let mut opts = ODEOptions::with_tolerances(1e-2, 1e-4);
        opts.max_steps = 50000;

        // Very stiff problem using dual operations
        let result = radau_impl(
            &client,
            |_t, y, c| dual_mul_scalar(y, -10000.0, c),
            [0.0, 0.01],
            &y0,
            &opts,
            &RadauOptions::<CpuRuntime>::default(),
        )
        .unwrap();

        assert!(
            result.success,
            "Radau should handle very stiff: {:?}",
            result.message
        );

        let y_final = result.y_final_vec();
        // exp(-10000 * 0.01) = exp(-100) ≈ 0
        // With relaxed tolerances, just check it's small
        assert!(
            y_final[0].abs() < 1e-6,
            "y_final = {}, should be close to 0",
            y_final[0]
        );
    }
}
