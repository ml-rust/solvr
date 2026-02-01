//! Multivariate unconstrained minimization.
//!
//! This module provides algorithms and configuration types for minimization.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod impl_generic;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use impl_generic::TensorMinimizeResult;
#[cfg(feature = "wgpu")]
pub use wgpu::*;

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
