//! Anderson mixing trait for fixed-point iteration acceleration.
use crate::DType;

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options specific to Anderson mixing.
#[derive(Debug, Clone)]
pub struct AndersonOptions {
    /// Base root finding options
    pub base: RootOptions,
    /// Number of previous iterates to store (history depth)
    pub m: usize,
    /// Mixing parameter alpha (0 < alpha <= 1)
    pub alpha: f64,
}

impl Default for AndersonOptions {
    fn default() -> Self {
        Self {
            base: RootOptions::default(),
            m: 5,
            alpha: 1.0,
        }
    }
}

/// Caller-reported stopping decision for an evaluated Anderson iterate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AndersonConvergence {
    /// Nonnegative metric reported in the result. It must be finite when converged.
    pub residual_norm: f64,
    /// Whether the caller's stopping conditions pass at this evaluated iterate.
    pub converged: bool,
}

/// Anderson mixing for accelerating fixed-point iterations.
///
/// Given g(x) such that the fixed point x* = g(x*), Anderson mixing
/// finds x* by combining previous iterates using least-squares to
/// minimize the residual F(x) = g(x) - x.
pub trait AndersonAlgorithms<R: Runtime<DType = DType>> {
    /// Solve F(x) = g(x) - x = 0 using Anderson mixing.
    ///
    /// # Arguments
    /// * `g` - The fixed-point function g: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Anderson mixing options
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<R>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<R>>
    where
        G: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Solve with caller-defined stopping conditions on the original tensor values.
    ///
    /// Calls `convergence(x, gx, fx)`, where `gx = g(x)` and `fx = gx - x`.
    /// Its decision replaces the built-in norm test. `options.base.tol` does not override it.
    /// Returns the checked `x`, not `gx`, when the callback accepts an iterate.
    /// The callback also runs after exhaustion, which always returns `converged = false`.
    /// A zero iteration budget still evaluates the map and final diagnostic once.
    ///
    /// # Errors
    /// Returns map, runtime, or convergence-callback errors with evaluation context.
    /// Rejects NaN or negative metrics and infinite metrics marked converged.
    fn anderson_with_convergence<G, K>(
        &self,
        g: G,
        x0: &Tensor<R>,
        options: &AndersonOptions,
        convergence: K,
    ) -> Result<RootTensorResult<R>>
    where
        G: Fn(&Tensor<R>) -> Result<Tensor<R>>,
        K: Fn(&Tensor<R>, &Tensor<R>, &Tensor<R>) -> Result<AndersonConvergence>;
}
