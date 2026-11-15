//! Multivariate root result.

use crate::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from tensor-based multivariate root finding.
#[derive(Debug, Clone)]
pub struct TensorRootResult<R: Runtime<DType = DType>> {
    /// The root found.
    pub x: Tensor<R>,
    /// Function value at root (should be near zero).
    pub fun: Tensor<R>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Norm of the residual.
    pub residual_norm: f64,
    /// Whether the method converged.
    pub converged: bool,
}
