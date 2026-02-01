//! Types for multivariate minimization algorithms.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for multivariate minimization.
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (function value change)
    pub f_tol: f64,
    /// Tolerance for convergence (argument change)
    pub x_tol: f64,
    /// Tolerance for gradient norm (gradient-based methods)
    pub g_tol: f64,
    /// Step size for finite difference gradient approximation
    pub eps: f64,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            f_tol: 1e-8,
            x_tol: 1e-8,
            g_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result of tensor-based minimization.
#[derive(Debug, Clone)]
pub struct TensorMinimizeResult<R: Runtime> {
    /// Solution vector.
    pub x: Tensor<R>,
    /// Function value at solution.
    pub fun: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}
