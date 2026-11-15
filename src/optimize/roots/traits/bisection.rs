//! Batched bracketed scalar roots.

use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Width tolerance and work limits for independent scalar roots.
#[derive(Debug, Clone)]
pub struct BatchedBisectionOptions {
    /// Nonnegative absolute width tolerance.
    pub atol: f64,
    /// Nonnegative relative width tolerance.
    pub rtol: f64,
    /// Maximum bisection rounds. Must be positive.
    pub max_iter: usize,
    /// Maximum upper-bound expansions. Zero disables expansion.
    pub max_expansions: usize,
    /// Finite expansion multiplier greater than one.
    pub expansion_factor: f64,
}

impl Default for BatchedBisectionOptions {
    fn default() -> Self {
        Self {
            atol: 1e-8,
            rtol: 1e-6,
            max_iter: 100,
            max_expansions: 0,
            expansion_factor: 2.0,
        }
    }
}

/// Final brackets and endpoint residuals, preserving input shape and dtype.
#[derive(Debug, Clone)]
pub struct BatchedBracketResult<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub f_lower: Tensor<R>,
    pub f_upper: Tensor<R>,
    /// Number of batched bisection rounds.
    pub iterations: usize,
    /// Number of batched expansion rounds.
    pub expansions: usize,
}

/// Solve independent elementwise scalar equations on the configured client.
pub trait BatchedBisectionAlgorithms<R: Runtime<DType = DType>> {
    /// Bisect nonempty F32 or F64 brackets independently, freezing converged lanes.
    ///
    /// `f` must evaluate independent, continuous elementwise functions with deterministic residuals.
    /// Every callback output must retain the input shape, dtype, and device.
    /// Residuals must be finite in active lanes. Increasing and decreasing functions are supported.
    /// Exact endpoint roots collapse the bracket, including degenerate intervals.
    /// Same-sign nonzero endpoints expand only the upper bound, preserving the lower bound.
    /// Expansion uses `lower + expansion_factor * (upper - lower)`.
    /// Convergence requires `upper - lower <= atol + rtol * max(abs(lower), abs(upper))`.
    /// Both tolerances must be finite and nonnegative, with at least one positive.
    /// Unrepresentable tolerances, overflow, and exhausted work limits return errors.
    fn bisect_batched<F>(
        &self,
        f: F,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        options: &BatchedBisectionOptions,
    ) -> OptimizeResult<BatchedBracketResult<R>>
    where
        F: Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>;
}
