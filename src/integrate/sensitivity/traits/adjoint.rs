//! Adjoint sensitivity analysis trait.
//!
//! Defines the interface for computing parameter gradients via backward
//! integration of the adjoint ODE.

use numr::autograd::Var;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::types::{SensitivityOptions, SensitivityResult};
use crate::integrate::ODEOptions;
use crate::integrate::error::IntegrateResult;

/// Trait for adjoint sensitivity analysis algorithms.
///
/// Computes gradients of a cost function J = g(y(T)) with respect to
/// parameters p, where y(t) satisfies dy/dt = f(t, y, p).
///
/// # Mathematical Background
///
/// For ODE: dy/dt = f(t, y, p) with cost J = g(y(T))
///
/// The adjoint method computes ∂J/∂p efficiently via:
/// 1. Forward pass: Solve ODE, store checkpoints
/// 2. Terminal condition: λ(T) = ∂g/∂y(T)
/// 3. Adjoint ODE: dλ/dt = -λᵀ · (∂f/∂y), integrated backward
/// 4. Parameter gradient: ∂J/∂p = ∫₀ᵀ λᵀ · (∂f/∂p) dt
///
/// # Memory Efficiency
///
/// The checkpointing approach trades computation for memory:
/// - O(n_checkpoints) memory instead of O(n_steps)
/// - Recomputes forward solution between checkpoints during backward pass
pub trait AdjointSensitivityAlgorithms<R: Runtime> {
    /// Compute parameter gradients using adjoint sensitivity analysis.
    ///
    /// # Arguments
    ///
    /// * `f` - ODE function f(t, y, p) returning dy/dt as a Var for autograd
    /// * `g` - Cost function g(y_final) returning scalar cost as a Var
    /// * `t_span` - Integration interval [t0, T]
    /// * `y0` - Initial condition
    /// * `p` - Parameters with respect to which to differentiate
    /// * `ode_opts` - Options for forward ODE integration
    /// * `sens_opts` - Options for sensitivity analysis
    ///
    /// # Returns
    ///
    /// `SensitivityResult` containing gradient ∂J/∂p and diagnostics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use numr::autograd::{Var, var_mul_scalar};
    ///
    /// // dy/dt = -k*y, y(0) = 1, J = y(T)²
    /// // Analytical: ∂J/∂k = -2*T*y(T)²
    ///
    /// let f = |t: &Var<R>, y: &Var<R>, p: &Var<R>, c: &C| {
    ///     // dy/dt = -p[0] * y
    ///     var_mul_scalar(&var_mul(p, y, c)?, -1.0, c)
    /// };
    ///
    /// let g = |y_final: &Var<R>, c: &C| {
    ///     // J = y²
    ///     var_mul(y_final, y_final, c)
    /// };
    ///
    /// let result = client.adjoint_sensitivity(
    ///     f, g, [0.0, 1.0], &y0, &params, &ode_opts, &sens_opts
    /// )?;
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn adjoint_sensitivity<F, G>(
        &self,
        f: F,
        g: G,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        p: &Tensor<R>,
        ode_opts: &ODEOptions,
        sens_opts: &SensitivityOptions,
    ) -> IntegrateResult<SensitivityResult<R>>
    where
        F: Fn(&Var<R>, &Var<R>, &Var<R>, &Self) -> Result<Var<R>>,
        G: Fn(&Var<R>, &Self) -> Result<Var<R>>;
}
