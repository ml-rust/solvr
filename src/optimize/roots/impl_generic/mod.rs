//! Tensor-based implementations of multivariate root finding algorithms.

mod broyden;
pub mod helpers;
mod levenberg_marquardt;
mod newton;

pub use broyden::broyden1_impl;
pub use helpers::{jacobian_forward_impl, jvp_impl};
pub use levenberg_marquardt::levenberg_marquardt_impl;
pub use newton::newton_system_impl;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from tensor-based multivariate root finding.
#[derive(Debug, Clone)]
pub struct TensorRootResult<R: Runtime> {
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
