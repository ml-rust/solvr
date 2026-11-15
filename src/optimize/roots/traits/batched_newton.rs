//! Independent scalar Newton iterations driven by caller-provided corrections.

use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Absolute correction tolerance and maximum callback evaluations.
#[derive(Debug, Clone)]
pub struct BatchedNewtonOptions {
    /// Positive finite tolerance representable in the input dtype.
    pub correction_tolerance: f64,
    /// Positive maximum number of batched correction evaluations.
    pub max_iter: usize,
}

impl Default for BatchedNewtonOptions {
    fn default() -> Self {
        Self {
            correction_tolerance: 1e-8,
            max_iter: 100,
        }
    }
}

/// Final iterates, last evaluated corrections, and exact zero/one lane masks.
#[derive(Debug, Clone)]
pub struct BatchedNewtonResult<R: Runtime<DType = DType>> {
    /// Last finite iterate retained for each lane.
    pub x: Tensor<R>,
    /// Last correction evaluated while each lane remained active, including nonfinite values.
    pub correction: Tensor<R>,
    /// Input-dtype mask for strict correction convergence before updating the iterate.
    pub converged: Tensor<R>,
    /// Input-dtype mask for nonfinite corrections or candidate iterates.
    pub nonfinite: Tensor<R>,
    /// Number of batched callback evaluations, including the final convergence check.
    pub iterations: usize,
}

/// Solve independent elementwise equations with caller-defined Newton corrections.
pub trait BatchedNewtonAlgorithms<R: Runtime<DType = DType>> {
    /// Evaluate corrections and update active lanes with `x_next = x - correction`.
    ///
    /// The callback must preserve input shape, dtype, and device and evaluate deterministic independent lanes.
    /// Strict `abs(correction) < correction_tolerance` freezes the current iterate before updating.
    /// Convergence certifies correction size only. It does not bound equation residuals or root error.
    /// Nonfinite corrections or candidate iterates freeze the last finite iterate independently.
    /// Frozen lanes retain their last evaluated correction and ignore later callback lane values.
    /// Exhausted lanes return both masks zero. Rounded stagnation never implies convergence.
    /// Callback errors and invalid metadata return errors, without a partial result.
    fn newton_batched<F>(
        &self,
        correction_callback: F,
        initial: &Tensor<R>,
        options: &BatchedNewtonOptions,
    ) -> OptimizeResult<BatchedNewtonResult<R>>
    where
        F: Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>;
}
