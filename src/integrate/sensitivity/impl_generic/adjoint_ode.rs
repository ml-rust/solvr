//! Adjoint sensitivity analysis implementation.
//!
//! Computes parameter gradients via backward integration of the adjoint ODE.
//! Uses checkpointing for memory efficiency.
//!
//! # Adjoint Formulation
//!
//! This implementation uses the **continuous adjoint** method:
//!
//! - The forward pass integrates dy/dt = f(t, y, p) using the chosen ODE solver and
//!   stores a checkpoint at every solver step, so the true forward state `y` is
//!   known at both endpoints of every interval.
//! - The backward pass integrates the adjoint ODE with one second-order Heun
//!   (explicit trapezoidal) step per checkpoint interval, evaluating VJPs via
//!   reverse-mode autograd (`vjp_with_params`) at the stored endpoint states.
//!   Because each interval is a single accurate solver step, `y` is never
//!   reconstructed by an unstable backward sweep. This backward integrator is
//!   completely independent of the forward solver — it only uses the stored
//!   checkpoint states and the user's Var-based ODE function.
//!
//! Because the backward pass is forward-solver-agnostic, supporting implicit
//! solvers (BDF, Radau, LSODA) for the forward pass requires only wrapping
//! the primal ODE function in the DualTensor interface those solvers expect
//! for Jacobian computation.
use crate::DType;

use numr::autograd::{DualTensor, Var, backward};
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::checkpointing::CheckpointManager;
use crate::common::jacobian::vjp_with_params;
use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::impl_generic::ode::ODEResultTensor;
use crate::integrate::impl_generic::ode::stiff_client::StiffSolverClient;
use crate::integrate::ode::{BDFOptions, LSODAOptions, ODEMethod, ODEOptions, RadauOptions};
use crate::integrate::sensitivity::traits::{SensitivityOptions, SensitivityResult};

/// Internal forward ODE function wrapper.
///
/// Wraps the user's Var-based ODE function into a Tensor-based function
/// for the forward integration pass.
struct ForwardWrapper<'a, R, C, F>
where
    R: Runtime<DType = DType>,
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
    R: Runtime<DType = DType>,
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
    R: Runtime<DType = DType>,
    C: StiffSolverClient<R>,
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
        .contiguous()?;

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
    R: Runtime<DType = DType>,
    C: StiffSolverClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    // Store initial checkpoint
    checkpoint_manager.add_checkpoint(t_span[0], y0.clone());

    // Create the tensor-based ODE function for the solver
    let f_tensor = |t: &Tensor<R>, y: &Tensor<R>| -> Result<Tensor<R>> { wrapper.eval(t, y) };

    // Tolerances for the implicit-solver forward pass.
    //
    // Implicit solvers (BDF, Radau, LSODA) require Newton iteration and an error
    // estimate that together impose a *stability* constraint on step size.  When the
    // user requests very tight tolerances (e.g. rtol=1e-8, atol=1e-10) for the ODE
    // solution, the error check rejects almost every step on a stiff equation,
    // causing the solver to exhaust `max_steps` after advancing only a tiny fraction
    // of the time interval.
    //
    // For the adjoint forward pass we do not need the forward trajectory to match
    // the user's requested ODE accuracy.  The backward adjoint integrator (Euler)
    // evaluates VJPs at checkpoint states; it only needs those states to be accurate
    // enough to recover the adjoint to the tolerance requested via `adjoint_rtol`.
    // A forward-trajectory error of 1e-4 in y introduces a comparable relative error
    // in the adjoint gradient — well below the typical 5% adjoint tolerance.
    //
    // Fix: use a "relaxed" forward tolerance for implicit solvers that is the looser
    // of the user's requested tolerance and a sensible stiff-solver default (1e-3 rtol,
    // 1e-6 atol).  Explicit solvers are not affected — they don't have the same
    // Newton/error-step constraint and routinely achieve 1e-8 accuracy.
    let implicit_opts = {
        let mut o = options.clone();
        // Implicit solvers (BDF/Radau/LSODA) may need many more steps than
        // explicit solvers to complete the full t_span, especially with tight
        // tolerances on stiff systems.  The default max_steps=10000 can be
        // exhausted before reaching t_end.
        //
        // For the adjoint forward pass we need the solver to cover the complete
        // time interval; the backward adjoint pass handles accuracy via its own
        // tolerances.  We therefore raise max_steps to 500 000 (a conservative
        // upper bound for stiff problems at rtol ≥ 1e-10) without changing the
        // requested ODE tolerances so that the forward trajectory accuracy is
        // exactly what the user specified.
        o.max_steps = o.max_steps.max(500_000);
        o
    };

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

        ODEMethod::BDF => {
            // Wrap the primal f_tensor in the DualTensor interface required by bdf_impl.
            //
            // bdf_impl computes the Jacobian ∂f/∂y by calling this closure with
            // unit-seed tangent vectors (column by column, via jacobian_forward).
            // If we return DualTensor::constant (tangent = None → zero), every
            // Jacobian column is zero → Newton iteration gets a singular matrix and
            // diverges → BDF takes infinitesimally small steps → forward trajectory
            // is wrong.
            //
            // Fix: propagate the tangent correctly using a finite-difference JVP:
            //   tangent_out ≈ (f(t, y + ε·v) − f(t, y)) / ε
            // where v = y_d.tangent() is the seed vector.  This is O(1) extra
            // f-evaluations per Jacobian column — identical to what a numerical-
            // Jacobian path would do anyway, just without a separate code path.
            let f_dual =
                |t_d: &DualTensor<R>, y_d: &DualTensor<R>, c: &C| -> Result<DualTensor<R>> {
                    let t_primal = t_d.primal();
                    let y_primal = y_d.primal();
                    let f_primal = f_tensor(t_primal, y_primal)?;

                    let tangent_out = if let Some(v) = y_d.tangent() {
                        // Finite-difference JVP: (f(y + ε·v) - f(y)) / ε
                        // ε chosen for double-precision FD accuracy (~1e-7).
                        let eps = 1e-7_f64;
                        let v_eps = c.mul_scalar(v, eps)?;
                        let y_pert = c.add(y_primal, &v_eps)?;
                        let f_pert = f_tensor(t_primal, &y_pert)?;
                        let diff = c.sub(&f_pert, &f_primal)?;
                        Some(c.mul_scalar(&diff, 1.0 / eps)?)
                    } else {
                        None
                    };

                    Ok(DualTensor::new(f_primal, tangent_out))
                };

            // Use a tight Newton tolerance so that Newton always performs at
            // least one correction step rather than "converging" immediately at
            // the predictor (which would degenerate to explicit Euler and give
            // zero error estimates that cause unlimited step-size growth).
            let bdf_opts =
                BDFOptions::default().newton_params((implicit_opts.atol * 1e-2).min(1e-10), 20);
            crate::integrate::impl_generic::ode::bdf_impl(
                client,
                f_dual,
                t_span,
                y0,
                &implicit_opts,
                &bdf_opts,
            )?
        }

        ODEMethod::Radau => {
            // Same JVP-via-finite-difference wrapping as BDF above.
            let f_dual =
                |t_d: &DualTensor<R>, y_d: &DualTensor<R>, c: &C| -> Result<DualTensor<R>> {
                    let t_primal = t_d.primal();
                    let y_primal = y_d.primal();
                    let f_primal = f_tensor(t_primal, y_primal)?;

                    let tangent_out = if let Some(v) = y_d.tangent() {
                        let eps = 1e-7_f64;
                        let v_eps = c.mul_scalar(v, eps)?;
                        let y_pert = c.add(y_primal, &v_eps)?;
                        let f_pert = f_tensor(t_primal, &y_pert)?;
                        let diff = c.sub(&f_pert, &f_primal)?;
                        Some(c.mul_scalar(&diff, 1.0 / eps)?)
                    } else {
                        None
                    };

                    Ok(DualTensor::new(f_primal, tangent_out))
                };

            let radau_opts =
                RadauOptions::default().newton_params((implicit_opts.atol * 1e-2).min(1e-10), 20);
            crate::integrate::impl_generic::ode::radau_impl(
                client,
                f_dual,
                t_span,
                y0,
                &implicit_opts,
                &radau_opts,
            )?
        }

        ODEMethod::LSODA => {
            // Same JVP-via-finite-difference wrapping as BDF above.
            let f_dual =
                |t_d: &DualTensor<R>, y_d: &DualTensor<R>, c: &C| -> Result<DualTensor<R>> {
                    let t_primal = t_d.primal();
                    let y_primal = y_d.primal();
                    let f_primal = f_tensor(t_primal, y_primal)?;

                    let tangent_out = if let Some(v) = y_d.tangent() {
                        let eps = 1e-7_f64;
                        let v_eps = c.mul_scalar(v, eps)?;
                        let y_pert = c.add(y_primal, &v_eps)?;
                        let f_pert = f_tensor(t_primal, &y_pert)?;
                        let diff = c.sub(&f_pert, &f_primal)?;
                        Some(c.mul_scalar(&diff, 1.0 / eps)?)
                    } else {
                        None
                    };

                    Ok(DualTensor::new(f_primal, tangent_out))
                };

            crate::integrate::impl_generic::ode::lsoda_impl(
                client,
                f_dual,
                t_span,
                y0,
                &implicit_opts,
                &LSODAOptions::default(),
            )?
        }

        // Verlet and Leapfrog are symplectic integrators for Hamiltonian systems
        // (separate q and p coordinates). They cannot be used as general ODE
        // solvers and have no meaningful forward trajectory for adjoint sensitivity.
        ODEMethod::Verlet | ODEMethod::Leapfrog => {
            return Err(IntegrateError::InvalidInput {
                context: format!(
                    "Symplectic method {:?} cannot be used for adjoint sensitivity: \
                     symplectic integrators require separate position/momentum coordinates \
                     (q, p) and are not general ODE solvers. Use RK45, RK23, DOP853, \
                     BDF, Radau, or LSODA instead.",
                    options.method
                ),
            });
        }
    };

    // Extract checkpoints from the solver result.
    //
    // Strategy: store every actual solver output step as a checkpoint.
    //
    // WHY: Explicit solvers (RK45) and implicit solvers (BDF/Radau/LSODA) both
    // use adaptive step sizes, so neither is guaranteed to land on the pre-planned
    // uniform checkpoint times.  The old code tried to match solver steps to those
    // planned times with a tight tolerance (1e-7) and stored a checkpoint only when
    // a match was found.  For stiff systems where BDF takes large steps that skip
    // over the planned times entirely, this left only 2 checkpoints (t0, tf).
    //
    // The backward pass then had to reconstruct y backward from y(T) using Euler
    // across the *entire* interval — completely wrong for stiff ODEs where the
    // solution changes rapidly and Euler backward reconstruction diverges.
    //
    // Fix: store ALL solver steps as checkpoints.  This gives the backward pass
    // fine-grained intervals where y is already known at both endpoints (from the
    // forward solve), so the backward reconstruction within each tiny interval is
    // accurate regardless of stiffness.  Memory cost is proportional to n_steps,
    // which is the minimum needed for correctness.
    //
    // Note: t_vec extraction is acceptable here (1D time array, post-solve API boundary).
    let t_vec: Vec<f64> = result.t.to_vec();
    let n_steps = t_vec.len();

    // Add all solver steps as checkpoints, skipping t0 (already stored) and
    // deduplicating steps that happen to coincide (within checkpoint_tol).
    let mut last_t = t_span[0]; // t0 is already stored
    for (idx, &t_val) in t_vec.iter().enumerate().take(n_steps) {
        // Skip the initial time (already stored) and duplicate times.
        if (t_val - last_t).abs() < checkpoint_tol {
            continue;
        }
        let y_checkpoint = result
            .y
            .narrow(0, idx, 1)
            .map_err(|e| IntegrateError::NumericalError {
                message: format!("Failed to extract checkpoint state at step {}: {}", idx, e),
            })?
            .squeeze(Some(0))
            .contiguous()?;
        checkpoint_manager.add_checkpoint(t_val, y_checkpoint);
        last_t = t_val;
    }

    // Guarantee the final state is the last checkpoint (solver may have stopped
    // exactly at tf or slightly before due to step-size rounding).
    if n_steps > 0 {
        let t_last = t_vec[n_steps - 1];
        if checkpoint_manager.checkpoints().last().map(|c| c.t) != Some(t_last) {
            let y_final = result
                .y
                .narrow(0, n_steps - 1, 1)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Failed to extract final checkpoint: {}", e),
                })?
                .squeeze(Some(0))
                .contiguous()?;
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
    R: Runtime<DType = DType>,
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
    R: Runtime<DType = DType>,
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
            client,
            f,
            p,
            &lambda,
            &ck_end.y,   // y at t_start (later time) — accurate forward state
            &ck_start.y, // y at t_end (earlier time) — accurate forward state
            t_start,
            t_end,
            sens_opts,
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

/// Integrate the adjoint ODE backward over a single checkpoint interval using
/// one second-order Heun (explicit trapezoidal) step.
///
/// The forward solver stores a checkpoint at *every* step, so each interval is a
/// single accurate solver step and we know the true forward state `y` at BOTH
/// endpoints (`y_start` at the later time `t_start`, `y_end` at the earlier time
/// `t_end`). We therefore never reconstruct `y` by an unstable backward Euler
/// sweep — we evaluate the VJPs directly at the known states. This is both
/// correct for stiff systems and cheap (2 VJP evaluations per interval).
///
/// Integrated quantities (backward, so `Δt = t_end - t_start < 0`):
/// - Adjoint:  dλ/dt = -(∂f/∂y)ᵀ · λ
/// - Gradient: dG/dt =  λᵀ · (∂f/∂p)   accumulated with the trapezoidal rule.
#[allow(clippy::too_many_arguments)]
fn integrate_adjoint_interval<R, C, F>(
    client: &C,
    f: &F,
    p: &Tensor<R>,
    lambda_start: &Tensor<R>,
    y_start: &Tensor<R>, // y at t_start (later time)
    y_end: &Tensor<R>,   // y at t_end (earlier time)
    t_start: f64,
    t_end: f64,
    _sens_opts: &SensitivityOptions,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, usize)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    // dλ/dt = -(∂f/∂y)ᵀ · λ
    let rhs = |t_val: f64,
               y_cur: &Tensor<R>,
               lam: &Tensor<R>|
     -> IntegrateResult<(Tensor<R>, Tensor<R>)> {
        let (_f_val, vjp_y, vjp_p) =
            vjp_with_params(client, f, t_val, y_cur, p, lam).map_err(|e| {
                IntegrateError::NumericalError {
                    message: format!("VJP computation failed at t={}: {}", t_val, e),
                }
            })?;
        let dlambda_dt =
            client
                .mul_scalar(&vjp_y, -1.0)
                .map_err(|e| IntegrateError::NumericalError {
                    message: format!("Scalar multiply failed: {}", e),
                })?;
        Ok((dlambda_dt, vjp_p))
    };

    let dt = t_end - t_start; // negative (backward in time)

    let map_err = |ctx: &'static str| {
        move |e: numr::error::Error| IntegrateError::NumericalError {
            message: format!("{}: {}", ctx, e),
        }
    };

    // Stage 1: derivatives at the later endpoint (t_start, y_start, λ_start).
    let (k1_lambda, vjp_p1) = rhs(t_start, y_start, lambda_start)?;

    // Predictor: λ_pred = λ_start + Δt · k1
    let lambda_pred = client
        .add(
            lambda_start,
            &client
                .mul_scalar(&k1_lambda, dt)
                .map_err(map_err("predictor scale"))?,
        )
        .map_err(map_err("predictor add"))?;

    // Stage 2: derivatives at the earlier endpoint (t_end, y_end, λ_pred).
    let (k2_lambda, vjp_p2) = rhs(t_end, y_end, &lambda_pred)?;

    // Corrector: λ_end = λ_start + (Δt/2)(k1 + k2)
    let k_sum = client
        .add(&k1_lambda, &k2_lambda)
        .map_err(map_err("corrector sum"))?;
    let lambda_end = client
        .add(
            lambda_start,
            &client
                .mul_scalar(&k_sum, dt * 0.5)
                .map_err(map_err("corrector scale"))?,
        )
        .map_err(map_err("corrector add"))?;

    // Trapezoidal gradient: ΔG = (|Δt|/2)(λᵀ∂f/∂p|start + λᵀ∂f/∂p|end).
    let vjp_p_sum = client
        .add(&vjp_p1, &vjp_p2)
        .map_err(map_err("gradient sum"))?;
    let gradient = client
        .mul_scalar(&vjp_p_sum, dt.abs() * 0.5)
        .map_err(map_err("gradient scale"))?;

    Ok((lambda_end, gradient, 2))
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

    /// Build the standard ODE function and cost for y' = -k*y, J = y(T)²,
    /// and compute adjoint gradient with the given ODEOptions.
    ///
    /// Returns (adjoint_gradient, analytical_gradient) as f64.
    fn run_exponential_decay_adjoint(ode_opts: ODEOptions) -> (f64, f64) {
        let (device, client) = setup();

        let t_span = [0.0, 1.0];
        let k_val = 0.5f64;
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[k_val], &[1], &device);

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

        let sens_opts = SensitivityOptions::default()
            .with_checkpoints(10)
            .with_adjoint_tolerances(1e-6, 1e-8);

        let result =
            adjoint_sensitivity_impl(&client, f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
                .expect("adjoint_sensitivity_impl should not return Err");

        let grad_val = result.gradient.to_vec::<f64>()[0];

        // Analytical: y(T) = exp(-k*T), J = exp(-2kT), dJ/dk = -2T*exp(-2kT)
        let t_final = 1.0f64;
        let y_analytical = (-k_val * t_final).exp();
        let grad_analytical = -2.0 * t_final * y_analytical * y_analytical;

        (grad_val, grad_analytical)
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

    /// Regression test: BDF no longer returns InvalidInput.
    ///
    /// Before this fix, ODEMethod::BDF in the forward pass returned
    /// `Err(IntegrateError::InvalidInput { ... })` immediately. This test
    /// asserts the call succeeds (no panic or Err on the method-dispatch path).
    #[test]
    fn test_bdf_no_longer_returns_invalid_input() {
        let ode_opts = ODEOptions::with_tolerances(1e-6, 1e-8).method(ODEMethod::BDF);
        // Just ensure it doesn't return Err — the gradient is checked in
        // test_bdf_adjoint_stiff_linear_ode with a stricter assertion.
        let (grad_val, _) = run_exponential_decay_adjoint(ode_opts);
        // Gradient should be finite (not NaN/Inf), which would indicate a crash path.
        assert!(
            grad_val.is_finite(),
            "BDF adjoint gradient should be finite, got {}",
            grad_val
        );
    }

    /// Regression test: Radau no longer returns InvalidInput.
    #[test]
    fn test_radau_no_longer_returns_invalid_input() {
        let ode_opts = ODEOptions::with_tolerances(1e-6, 1e-8).method(ODEMethod::Radau);
        let (grad_val, _) = run_exponential_decay_adjoint(ode_opts);
        assert!(
            grad_val.is_finite(),
            "Radau adjoint gradient should be finite, got {}",
            grad_val
        );
    }

    /// Regression test: LSODA no longer returns InvalidInput.
    #[test]
    fn test_lsoda_no_longer_returns_invalid_input() {
        let ode_opts = ODEOptions::with_tolerances(1e-6, 1e-8).method(ODEMethod::LSODA);
        let (grad_val, _) = run_exponential_decay_adjoint(ode_opts);
        assert!(
            grad_val.is_finite(),
            "LSODA adjoint gradient should be finite, got {}",
            grad_val
        );
    }

    /// BDF adjoint gradient on a stiff linear ODE.
    ///
    /// ODE: dy/dt = -k*y, k = 50 (stiff for explicit methods, step constraint ~1e-4).
    /// Cost: J = y(T)², T = 0.1.
    /// Analytical: dJ/dk = -2T * exp(-2kT).
    ///
    /// The gradient is also cross-checked against a central finite-difference
    /// estimate using BDF forward passes, so the test validates both that the
    /// adjoint method gives correct gradients and that BDF integrates correctly.
    #[test]
    fn test_bdf_adjoint_stiff_linear_ode() {
        let (device, client) = setup();

        let k_val = 50.0f64;
        let t_span = [0.0, 0.1];
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[k_val], &[1], &device);

        // ODE: dy/dt = -k * y  (stiff for large k)
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

        let ode_opts = ODEOptions::with_tolerances(1e-8, 1e-10).method(ODEMethod::BDF);
        let sens_opts = SensitivityOptions::default()
            .with_checkpoints(20)
            .with_adjoint_tolerances(1e-6, 1e-8);

        let result =
            adjoint_sensitivity_impl(&client, f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
                .expect("BDF adjoint should succeed on stiff linear ODE");

        let adjoint_grad = result.gradient.to_vec::<f64>()[0];

        // Analytical gradient: dJ/dk = -2T * exp(-2kT)
        let t_final = t_span[1];
        let y_analytical = (-k_val * t_final).exp();
        let grad_analytical = -2.0 * t_final * y_analytical * y_analytical;

        // 5% relative tolerance: the continuous adjoint backward pass uses
        // fixed-step Euler, so some numerical error is expected vs. analytical.
        let rel_err = (adjoint_grad - grad_analytical).abs() / grad_analytical.abs();
        assert!(
            rel_err < 0.05,
            "BDF adjoint gradient: expected {:.6e}, got {:.6e} (rel error {:.2}%)",
            grad_analytical,
            adjoint_grad,
            rel_err * 100.0
        );

        // Cross-check: finite-difference estimate of dJ/dk using two BDF forward passes.
        // This validates the BDF forward trajectory is correct independently of the
        // adjoint backward pass.
        let eps = 1e-4;
        let k_plus = Tensor::<CpuRuntime>::from_slice(&[k_val + eps], &[1], &device);
        let k_minus = Tensor::<CpuRuntime>::from_slice(&[k_val - eps], &[1], &device);

        let f_for_fd = |_t: &Var<CpuRuntime>,
                        y: &Var<CpuRuntime>,
                        p: &Var<CpuRuntime>,
                        c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };
        let g_for_fd =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let ode_opts_fd = ODEOptions::with_tolerances(1e-10, 1e-12).method(ODEMethod::BDF);
        let sens_opts_fd = SensitivityOptions::default().with_checkpoints(5);

        let res_plus = adjoint_sensitivity_impl(
            &client,
            f_for_fd,
            g_for_fd,
            t_span,
            &y0,
            &k_plus,
            &ode_opts_fd,
            &sens_opts_fd,
        )
        .expect("BDF adjoint (k+eps) should succeed");

        let f_for_fd2 = |_t: &Var<CpuRuntime>,
                         y: &Var<CpuRuntime>,
                         p: &Var<CpuRuntime>,
                         c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };
        let g_for_fd2 =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let res_minus = adjoint_sensitivity_impl(
            &client,
            f_for_fd2,
            g_for_fd2,
            t_span,
            &y0,
            &k_minus,
            &ode_opts_fd,
            &sens_opts_fd,
        )
        .expect("BDF adjoint (k-eps) should succeed");

        let fd_grad = (res_plus.cost - res_minus.cost) / (2.0 * eps);

        // FD gradient vs analytical should be very tight
        let fd_err = (fd_grad - grad_analytical).abs() / grad_analytical.abs();
        assert!(
            fd_err < 1e-3,
            "BDF finite-difference gradient: expected {:.6e}, got {:.6e} (rel error {:.4}%)",
            grad_analytical,
            fd_grad,
            fd_err * 100.0
        );

        // Adjoint gradient vs FD gradient (additional cross-check)
        let adj_fd_err = (adjoint_grad - fd_grad).abs() / fd_grad.abs();
        assert!(
            adj_fd_err < 0.05,
            "BDF adjoint vs FD: adjoint = {:.6e}, fd = {:.6e} (rel error {:.2}%)",
            adjoint_grad,
            fd_grad,
            adj_fd_err * 100.0
        );
    }

    /// Helper to call adjoint_sensitivity_impl with a simple exponential decay ODE
    /// and return only whether it errors.
    fn adjoint_with_method(method: ODEMethod) -> bool {
        let (device, client) = setup();
        let t_span = [0.0, 1.0];
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);

        let f = |_t: &Var<CpuRuntime>,
                 y: &Var<CpuRuntime>,
                 p: &Var<CpuRuntime>,
                 c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };
        let g =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let ode_opts = ODEOptions::with_method(method);
        let sens_opts = SensitivityOptions::default().with_checkpoints(5);

        adjoint_sensitivity_impl(&client, f, g, t_span, &y0, &k, &ode_opts, &sens_opts).is_err()
    }

    /// Helper to get the error message string when adjoint_sensitivity_impl returns Err.
    fn adjoint_error_msg(method: ODEMethod) -> String {
        let (device, client) = setup();
        let t_span = [0.0, 1.0];
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);

        let f = |_t: &Var<CpuRuntime>,
                 y: &Var<CpuRuntime>,
                 p: &Var<CpuRuntime>,
                 c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };
        let g =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let ode_opts = ODEOptions::with_method(method);
        let sens_opts = SensitivityOptions::default().with_checkpoints(5);

        format!(
            "{:?}",
            adjoint_sensitivity_impl(&client, f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
                .unwrap_err()
        )
    }

    /// Verlet returns a clear error — symplectic methods cannot be used as general
    /// ODE solvers for adjoint sensitivity.
    #[test]
    fn test_verlet_returns_meaningful_error() {
        assert!(
            adjoint_with_method(ODEMethod::Verlet),
            "Verlet should return Err for adjoint sensitivity"
        );
        let msg = adjoint_error_msg(ODEMethod::Verlet);
        assert!(
            msg.contains("symplectic") || msg.contains("Verlet"),
            "Error should mention symplectic nature: {}",
            msg
        );
    }

    /// Leapfrog returns a clear error — same reason as Verlet.
    #[test]
    fn test_leapfrog_returns_meaningful_error() {
        assert!(
            adjoint_with_method(ODEMethod::Leapfrog),
            "Leapfrog should return Err for adjoint sensitivity"
        );
        let msg = adjoint_error_msg(ODEMethod::Leapfrog);
        assert!(
            msg.contains("symplectic") || msg.contains("Leapfrog"),
            "Error should mention symplectic nature: {}",
            msg
        );
    }
}
