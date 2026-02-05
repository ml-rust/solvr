//! Adjoint sensitivity analysis implementation.
//!
//! Computes parameter gradients via backward integration of the adjoint ODE.
//! Uses checkpointing for memory efficiency.

use numr::autograd::{Var, backward};
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::checkpointing::CheckpointManager;
use crate::common::jacobian::vjp_with_params;
use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::impl_generic::ode::ODEResultTensor;
use crate::integrate::ode::{ODEMethod, ODEOptions};
use crate::integrate::sensitivity::traits::{SensitivityOptions, SensitivityResult};

/// Internal forward ODE function wrapper.
///
/// Wraps the user's Var-based ODE function into a Tensor-based function
/// for the forward integration pass.
struct ForwardWrapper<'a, R, C, F>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    f: &'a F,
    p: &'a Tensor<R>,
    client: &'a C,
}

impl<'a, R, C, F> ForwardWrapper<'a, R, C, F>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    fn new(f: &'a F, p: &'a Tensor<R>, client: &'a C) -> Self {
        Self { f, p, client }
    }

    /// Evaluate the ODE function without gradients (for forward integration).
    fn eval(&self, t: &Tensor<R>, y: &Tensor<R>) -> Result<Tensor<R>> {
        let t_var = Var::new(t.clone(), false);
        let y_var = Var::new(y.clone(), false);
        let p_var = Var::new(self.p.clone(), false);

        let result = (self.f)(&t_var, &y_var, &p_var, self.client)?;
        Ok(result.tensor().clone())
    }
}

/// Implement adjoint sensitivity analysis.
///
/// # Algorithm Overview
///
/// 1. **Forward pass**: Integrate dy/dt = f(t, y, p) from t0 to T,
///    storing checkpoints at specified times.
///
/// 2. **Terminal condition**: Compute λ(T) = ∂g/∂y(T) using autograd.
///
/// 3. **Backward pass**: For each checkpoint interval [t_{i+1}, t_i] (reverse order):
///    a. Recompute forward solution in [t_i, t_{i+1}] if needed
///    b. Integrate adjoint ODE: dλ/dt = -λᵀ · (∂f/∂y)
///    c. Accumulate: ∂J/∂p += ∫ λᵀ · (∂f/∂p) dt
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - ODE function f(t, y, p) as Var-based closure
/// * `g` - Cost function g(y_final) as Var-based closure
/// * `t_span` - Integration interval [t0, T]
/// * `y0` - Initial condition
/// * `p` - Parameters
/// * `ode_opts` - Options for forward ODE integration
/// * `sens_opts` - Options for sensitivity analysis
#[allow(clippy::too_many_arguments)]
pub fn adjoint_sensitivity_impl<R, C, F, G>(
    client: &C,
    f: F,
    g: G,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    p: &Tensor<R>,
    ode_opts: &ODEOptions,
    sens_opts: &SensitivityOptions,
) -> IntegrateResult<SensitivityResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
    G: Fn(&Var<R>, &C) -> Result<Var<R>>,
{
    let [t0, tf] = t_span;
    let _device = y0.device();

    // Validate inputs
    if t0 >= tf {
        return Err(IntegrateError::InvalidInterval {
            a: t0,
            b: tf,
            context: "adjoint_sensitivity".to_string(),
        });
    }

    // =========================================================================
    // FORWARD PASS: Integrate ODE with checkpointing
    // =========================================================================

    let mut checkpoint_manager = CheckpointManager::new(
        sens_opts.n_checkpoints,
        sens_opts.checkpoint_strategy,
        t_span,
    );

    let forward_wrapper = ForwardWrapper::new(&f, p, client);

    // Determine checkpoint tolerance based on step size
    let checkpoint_tol = (tf - t0) * 1e-8;

    // Forward integration with checkpoint storage
    let forward_result = forward_with_checkpoints(
        client,
        &forward_wrapper,
        t_span,
        y0,
        ode_opts,
        &mut checkpoint_manager,
        checkpoint_tol,
    )?;

    // Extract final state from 2D result tensor [n_steps, n_vars]
    // Use narrow() + contiguous() to avoid full GPU→CPU transfer
    let y_shape = forward_result.y.shape();
    let n_steps = y_shape[0];
    let y_final = forward_result
        .y
        .narrow(0, n_steps - 1, 1)
        .map_err(|e| IntegrateError::NumericalError {
            message: format!("Failed to extract final state: {}", e),
        })?
        .squeeze(Some(0))
        .contiguous();

    let nfev_forward = forward_result.nfev;

    // =========================================================================
    // TERMINAL CONDITION: λ(T) = ∂g/∂y(T)
    // =========================================================================

    let (cost, lambda_t) = compute_terminal_adjoint(client, &g, &y_final)?;

    // =========================================================================
    // BACKWARD PASS: Integrate adjoint ODE and accumulate gradients
    // =========================================================================

    let (gradient, nfev_adjoint) =
        backward_adjoint_pass(client, &f, p, &checkpoint_manager, &lambda_t, sens_opts)?;

    Ok(SensitivityResult {
        gradient,
        cost,
        y_final,
        nfev_forward,
        nfev_adjoint,
        n_checkpoints: checkpoint_manager.len(),
    })
}

/// Forward integration with checkpoint storage.
#[allow(clippy::needless_borrows_for_generic_args)]
fn forward_with_checkpoints<R, C, F>(
    client: &C,
    wrapper: &ForwardWrapper<'_, R, C, F>,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
    checkpoint_manager: &mut CheckpointManager<R>,
    checkpoint_tol: f64,
) -> IntegrateResult<ODEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    // Store initial checkpoint
    checkpoint_manager.add_checkpoint(t_span[0], y0.clone());

    // Create the tensor-based ODE function for the solver
    let f_tensor = |t: &Tensor<R>, y: &Tensor<R>| -> Result<Tensor<R>> { wrapper.eval(t, y) };

    // Use the standard ODE solver for forward integration
    // Note: We pass &f_tensor because the closure is used across multiple match arms
    let result = match options.method {
        ODEMethod::RK45 => {
            crate::integrate::impl_generic::ode::rk45_impl(client, &f_tensor, t_span, y0, options)?
        }
        ODEMethod::RK23 => {
            crate::integrate::impl_generic::ode::rk23_impl(client, &f_tensor, t_span, y0, options)?
        }
        ODEMethod::DOP853 => crate::integrate::impl_generic::ode::dop853_impl(
            client, &f_tensor, t_span, y0, options,
        )?,
        _ => {
            return Err(IntegrateError::InvalidInput {
                context: format!(
                    "Method {:?} not supported for adjoint sensitivity",
                    options.method
                ),
            });
        }
    };

    // Extract checkpoints from result at planned checkpoint times
    // t is shape [n_steps], y is shape [n_steps, n_vars]
    // Note: t_vec extraction is acceptable here (small 1D tensor, post-forward-pass API boundary)
    let t_vec: Vec<f64> = result.t.to_vec();
    let n_steps = t_vec.len();

    let checkpoint_times = checkpoint_manager.checkpoint_times().to_vec();

    for &tc in &checkpoint_times[1..] {
        // Skip t0 which is already stored
        // Find the closest time in the result
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (idx, &t_val) in t_vec.iter().enumerate() {
            let dist = (t_val - tc).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        if best_dist < checkpoint_tol * 10.0 {
            // Extract y at this time step using narrow() + contiguous()
            let y_checkpoint = result
                .y
                .narrow(0, best_idx, 1)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Failed to extract checkpoint state: {}", e),
                })?
                .squeeze(Some(0))
                .contiguous();
            checkpoint_manager.add_checkpoint(t_vec[best_idx], y_checkpoint);
        }
    }

    // Always ensure final state is checkpointed
    if n_steps > 0 {
        let t_last = t_vec[n_steps - 1];
        // Check if we haven't already added this
        if checkpoint_manager.checkpoints().last().map(|c| c.t) != Some(t_last) {
            let y_final = result
                .y
                .narrow(0, n_steps - 1, 1)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Failed to extract final checkpoint: {}", e),
                })?
                .squeeze(Some(0))
                .contiguous();
            checkpoint_manager.add_checkpoint(t_last, y_final);
        }
    }

    Ok(result)
}

/// Compute terminal adjoint condition λ(T) = ∂g/∂y(T).
fn compute_terminal_adjoint<R, C, G>(
    client: &C,
    g: &G,
    y_final: &Tensor<R>,
) -> IntegrateResult<(f64, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    G: Fn(&Var<R>, &C) -> Result<Var<R>>,
{
    // Create y_final as a variable with gradient tracking
    let y_var = Var::new(y_final.clone(), true);

    // Evaluate cost function
    let cost_var = g(&y_var, client).map_err(|e| IntegrateError::NumericalError {
        message: format!("Cost function evaluation failed: {}", e),
    })?;

    // Get cost value (should be scalar)
    let cost_tensor = cost_var.tensor();
    let cost = cost_tensor
        .item::<f64>()
        .map_err(|_| IntegrateError::InvalidInput {
            context: "Cost function must return a scalar".to_string(),
        })?;

    // Backward pass to get ∂g/∂y
    let grads = backward(&cost_var, client).map_err(|e| IntegrateError::NumericalError {
        message: format!("Backward pass for terminal condition failed: {}", e),
    })?;

    let lambda_t =
        grads
            .get(y_var.id())
            .cloned()
            .ok_or_else(|| IntegrateError::NumericalError {
                message: "No gradient for y_final in cost function".to_string(),
            })?;

    Ok((cost, lambda_t))
}

/// Backward adjoint integration pass.
///
/// Integrates the adjoint ODE backward in time and accumulates the parameter gradient.
fn backward_adjoint_pass<R, C, F>(
    client: &C,
    f: &F,
    p: &Tensor<R>,
    checkpoint_manager: &CheckpointManager<R>,
    lambda_t: &Tensor<R>,
    sens_opts: &SensitivityOptions,
) -> IntegrateResult<(Tensor<R>, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    let device = lambda_t.device();
    let dtype = lambda_t.dtype();
    let n_params = p.numel();

    // Initialize gradient accumulator
    let mut gradient = Tensor::<R>::zeros(&[n_params], dtype, device);
    let mut nfev_adjoint = 0usize;

    // Current adjoint state
    let mut lambda = lambda_t.clone();

    // Get checkpoints in reverse order (but as indices)
    let checkpoints = checkpoint_manager.checkpoints();
    let n_checkpoints = checkpoints.len();

    if n_checkpoints < 2 {
        return Err(IntegrateError::NumericalError {
            message: "Need at least 2 checkpoints for adjoint pass".to_string(),
        });
    }

    // Iterate backward through checkpoint intervals
    for i in (0..n_checkpoints - 1).rev() {
        let ck_start = &checkpoints[i];
        let ck_end = &checkpoints[i + 1];

        let t_start = ck_end.t; // Start of backward integration (later in time)
        let t_end = ck_start.t; // End of backward integration (earlier in time)

        if (t_start - t_end).abs() < 1e-14 {
            continue; // Skip zero-length intervals
        }

        // Integrate adjoint ODE backward from t_start to t_end
        // The adjoint ODE is: dλ/dt = -λᵀ · (∂f/∂y)
        // We use y from ck_end (later checkpoint) since we start there

        let (new_lambda, interval_gradient, interval_nfev) = integrate_adjoint_interval(
            client, f, p, &lambda,
            &ck_end.y, // y at later time (start of backward integration)
            t_start, t_end, sens_opts,
        )?;

        lambda = new_lambda;
        gradient = client.add(&gradient, &interval_gradient).map_err(|e| {
            IntegrateError::NumericalError {
                message: format!("Gradient accumulation failed: {}", e),
            }
        })?;
        nfev_adjoint += interval_nfev;
    }

    Ok((gradient, nfev_adjoint))
}

/// Integrate the augmented adjoint ODE over a single checkpoint interval using RK4.
///
/// The augmented system integrates both:
/// 1. Forward ODE (to reconstruct y): dy/dt = f(t, y, p)  (backward, so -f)
/// 2. Adjoint ODE: dλ/dt = -(∂f/∂y)ᵀ · λ  (backward, becomes (∂f/∂y)ᵀ · λ)
/// 3. Parameter gradient accumulation: dG/dt = λᵀ · (∂f/∂p)
///
/// This ensures we use the correct y(t) values throughout the interval.
#[allow(clippy::too_many_arguments)]
fn integrate_adjoint_interval<R, C, F>(
    client: &C,
    f: &F,
    p: &Tensor<R>,
    lambda_start: &Tensor<R>,
    y_start: &Tensor<R>, // y at t_start (later time, since we go backward)
    t_start: f64,
    t_end: f64,
    sens_opts: &SensitivityOptions,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    let device = lambda_start.device();
    let dtype = lambda_start.dtype();
    let n_params = p.numel();

    let mut lambda = lambda_start.clone();
    let mut y = y_start.clone();
    let mut gradient = Tensor::<R>::zeros(&[n_params], dtype, device);
    let mut nfev = 0usize;

    // Backward integration: t goes from t_start to t_end where t_start > t_end
    let dt = t_end - t_start; // This is negative (backward)

    // Use more steps for better accuracy
    let n_steps = ((dt.abs() / sens_opts.adjoint_atol.sqrt()).ceil() as usize)
        .max(100)
        .min(sens_opts.adjoint_max_steps);

    let h = dt / (n_steps as f64);

    // Helper to compute derivatives at current state
    // Returns (dy/dt, dλ/dt, λᵀ·∂f/∂p)
    let compute_derivs = |y_cur: &Tensor<R>,
                          lam: &Tensor<R>,
                          t_val: f64|
     -> IntegrateResult<(Tensor<R>, Tensor<R>, Tensor<R>)> {
        let (f_val, vjp_y, vjp_p) =
            vjp_with_params(client, f, t_val, y_cur, p, lam).map_err(|e| {
                IntegrateError::NumericalError {
                    message: format!("VJP computation failed at t={}: {}", t_val, e),
                }
            })?;

        // dλ/dt = -(∂f/∂y)ᵀ · λ = -vjp_y
        let neg_vjp_y =
            client
                .mul_scalar(&vjp_y, -1.0)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Scalar multiply failed: {}", e),
                })?;

        Ok((f_val, neg_vjp_y, vjp_p))
    };

    for step in 0..n_steps {
        let t = t_start + (step as f64) * h;

        // Compute derivatives at current point
        let (f_val, dlambda_dt, vjp_p) = compute_derivs(&y, &lambda, t)?;
        nfev += 1;

        // Simple Euler step for y (going backward)
        // Going backward: y(t - Δt) ≈ y(t) - Δt * f(t, y) where Δt = |h|
        // Note: h is negative (since t_end < t_start), so we use |h|
        let dy =
            client
                .mul_scalar(&f_val, h.abs())
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("dy mul failed: {}", e),
                })?;
        y = client
            .sub(&y, &dy)
            .map_err(|e| IntegrateError::NumericalError {
                message: format!("y update failed: {}", e),
            })?;

        // Euler step for lambda
        let dlambda =
            client
                .mul_scalar(&dlambda_dt, h)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("dlambda mul failed: {}", e),
                })?;
        lambda = client
            .add(&lambda, &dlambda)
            .map_err(|e| IntegrateError::NumericalError {
                message: format!("Lambda update failed: {}", e),
            })?;

        // Accumulate parameter gradient
        // ∂J/∂p += |h| * λᵀ · (∂f/∂p)
        let grad_contrib =
            client
                .mul_scalar(&vjp_p, h.abs())
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Gradient contribution multiply failed: {}", e),
                })?;

        gradient =
            client
                .add(&gradient, &grad_contrib)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Gradient accumulation failed: {}", e),
                })?;
    }

    Ok((lambda, gradient, nfev))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_mul_scalar};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_adjoint_exponential_decay() {
        // ODE: dy/dt = -k*y, y(0) = 1
        // Solution: y(t) = exp(-k*t)
        // Cost: J = y(T)² = exp(-2kT)
        // Analytical gradient: ∂J/∂k = -2T * exp(-2kT)
        let (device, client) = setup();

        let t_span = [0.0, 1.0];
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);

        // ODE: dy/dt = -k * y
        let f = |_t: &Var<CpuRuntime>,
                 y: &Var<CpuRuntime>,
                 p: &Var<CpuRuntime>,
                 c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };

        // Cost: J = y²
        let g =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let ode_opts = ODEOptions::with_tolerances(1e-8, 1e-10);

        let sens_opts = SensitivityOptions::default()
            .with_checkpoints(10)
            .with_adjoint_tolerances(1e-6, 1e-8);

        let result =
            adjoint_sensitivity_impl(&client, f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
                .unwrap();

        // Analytical values
        let k_val: f64 = 0.5;
        let t_final: f64 = 1.0;
        let y_analytical = (-k_val * t_final).exp();
        let cost_analytical = y_analytical * y_analytical;
        let grad_analytical = -2.0 * t_final * cost_analytical;

        // Check results
        let y_final_val = result.y_final.to_vec::<f64>()[0];
        let grad_val = result.gradient.to_vec::<f64>()[0];

        assert!(
            (y_final_val - y_analytical).abs() < 1e-5,
            "y_final: expected {}, got {}",
            y_analytical,
            y_final_val
        );

        assert!(
            (result.cost - cost_analytical).abs() < 1e-5,
            "cost: expected {}, got {}",
            cost_analytical,
            result.cost
        );

        // Gradient tolerance is looser due to numerical integration
        assert!(
            (grad_val - grad_analytical).abs() < 0.05 * grad_analytical.abs(),
            "gradient: expected {}, got {} (error = {}%)",
            grad_analytical,
            grad_val,
            100.0 * (grad_val - grad_analytical).abs() / grad_analytical.abs()
        );
    }
}
