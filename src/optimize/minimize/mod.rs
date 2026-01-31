//! Multivariate unconstrained minimization algorithms.
//!
//! This module provides methods for finding minima of scalar-valued functions
//! f: R^n -> R without constraints.

mod bfgs;
mod conjugate_gradient;
mod nelder_mead;
mod powell;

pub use bfgs::bfgs;
pub use conjugate_gradient::conjugate_gradient;
pub use nelder_mead::nelder_mead;
pub use powell::powell;

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

/// Result from a multivariate minimization method.
#[derive(Debug, Clone)]
pub struct MultiMinimizeResult {
    /// The minimum point found
    pub x: Vec<f64>,
    /// Function value at minimum
    pub fun: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_compare_methods() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let x0 = &[0.0, 0.0];

        let nm_result = nelder_mead(&f, x0, &MinimizeOptions::default()).expect("nm failed");
        let powell_result = powell(&f, x0, &MinimizeOptions::default()).expect("powell failed");
        let bfgs_result = bfgs(&f, x0, &MinimizeOptions::default()).expect("bfgs failed");
        let cg_result = conjugate_gradient(&f, x0, &MinimizeOptions::default()).expect("cg failed");

        // All should find minimum at (3, -1)
        assert!((nm_result.x[0] - 3.0).abs() < 1e-3);
        assert!((powell_result.x[0] - 3.0).abs() < 1e-3);
        assert!((bfgs_result.x[0] - 3.0).abs() < 1e-4);
        assert!((cg_result.x[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_empty_input() {
        let result = nelder_mead(sphere, &[], &MinimizeOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }
}
