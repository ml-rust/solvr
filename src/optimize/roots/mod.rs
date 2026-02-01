//! Multivariate root finding algorithms.
//!
//! This module provides methods for finding roots of systems of nonlinear equations.
//! Given F: R^n -> R^n, find x such that F(x) = 0.
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! roots/
//! ├── mod.rs    # Trait definition + types (exports only)
//! ├── cpu.rs    # CPU impl + scalar convenience functions
//! ├── cuda.rs   # CUDA impl (pure delegation)
//! └── wgpu.rs   # WebGPU impl (pure delegation)
//! ```
//!
//! Generic implementations live in `optimize/impl_generic/roots/`.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export CPU implementation for convenience
pub use cpu::*;

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct RootOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (norm of F(x))
    pub tol: f64,
    /// Tolerance for step size
    pub x_tol: f64,
    /// Step size for finite difference Jacobian approximation
    pub eps: f64,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            x_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result from a multivariate root finding method (scalar API).
#[derive(Debug, Clone)]
pub struct MultiRootResult {
    /// The root found
    pub x: Vec<f64>,
    /// Function value at root (should be near zero)
    pub fun: Vec<f64>,
    /// Number of iterations used
    pub iterations: usize,
    /// Norm of the residual
    pub residual_norm: f64,
    /// Whether the method converged
    pub converged: bool,
}

/// Algorithmic contract for root finding operations.
///
/// All backends implementing root finding MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait RootFindingAlgorithms<R: Runtime> {
    /// Newton's method for systems of nonlinear equations.
    ///
    /// Uses finite differences to approximate the Jacobian.
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Broyden's method (rank-1 update) for systems of nonlinear equations.
    ///
    /// A quasi-Newton method that approximates the Jacobian using rank-1 updates.
    fn broyden1<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Levenberg-Marquardt algorithm for systems of nonlinear equations.
    ///
    /// A damped Newton method that interpolates between Newton's method and
    /// gradient descent. More robust when initial guess is far from solution.
    fn levenberg_marquardt<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}

/// Result from tensor-based root finding.
#[derive(Debug, Clone)]
pub struct RootTensorResult<R: Runtime> {
    /// The root found
    pub x: Tensor<R>,
    /// Function value at root
    pub fun: Tensor<R>,
    /// Number of iterations used
    pub iterations: usize,
    /// Norm of the residual
    pub residual_norm: f64,
    /// Whether the method converged
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = RootOptions::default();
        assert_eq!(opts.max_iter, 100);
        assert!((opts.tol - 1e-8).abs() < 1e-12);
    }
}
