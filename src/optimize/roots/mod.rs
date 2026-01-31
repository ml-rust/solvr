//! Multivariate root finding algorithms.
//!
//! This module provides methods for finding roots of systems of nonlinear equations.
//! Given F: R^n -> R^n, find x such that F(x) = 0.

#![allow(clippy::needless_range_loop)]

mod quasi_newton;

pub use quasi_newton::{broyden1, levenberg_marquardt};

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{finite_difference_jacobian, norm, solve_linear_system};

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

/// Result from a multivariate root finding method.
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

/// Newton's method for systems of nonlinear equations.
///
/// # Arguments
/// * `f` - Function F: R^n -> R^n to find root of
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `F` (where F(x) ≈ 0)
///
/// # Note
/// Uses finite differences to approximate the Jacobian.
/// Has quadratic convergence near the root but may diverge if x0 is far from root.
pub fn newton_system<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "newton_system: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "newton_system: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    for iter in 0..options.max_iter {
        let res_norm = norm(&fx);

        if res_norm < options.tol {
            return Ok(MultiRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        let jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);

        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jacobian, &neg_fx) {
            Some(dx) => dx,
            None => {
                return Err(OptimizeError::NumericalError {
                    message: "Singular Jacobian in newton_system".to_string(),
                });
            }
        };

        for i in 0..n {
            x[i] += dx[i];
        }

        if norm(&dx) < options.x_tol {
            fx = f(&x);
            return Ok(MultiRootResult {
                x,
                fun: fx.clone(),
                iterations: iter + 1,
                residual_norm: norm(&fx),
                converged: true,
            });
        }

        fx = f(&x);
    }

    Ok(MultiRootResult {
        x,
        fun: fx.clone(),
        iterations: options.max_iter,
        residual_norm: norm(&fx),
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_system_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result =
            newton_system(f, &[0.0, 0.0], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_quadratic() {
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];

        let result =
            newton_system(f, &[0.5, 0.5], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        let expected = 1.0 / (2.0_f64).sqrt();
        assert!((result.x[0] - expected).abs() < 1e-6);
        assert!((result.x[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_3d() {
        let f = |x: &[f64]| vec![x[0] + x[1] + x[2] - 6.0, x[0] - x[1], x[1] - x[2]];

        let result = newton_system(f, &[1.0, 1.0, 1.0], &RootOptions::default())
            .expect("newton_system failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
        assert!((result.x[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_empty_input() {
        let f = |_: &[f64]| vec![];
        let result = newton_system(f, &[], &RootOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_compare_methods() {
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1]];

        let newton_result =
            newton_system(&f, &[1.0, 1.0], &RootOptions::default()).expect("newton failed");
        let broyden_result =
            broyden1(&f, &[1.0, 1.0], &RootOptions::default()).expect("broyden failed");
        let lm_result =
            levenberg_marquardt(&f, &[1.0, 1.0], &RootOptions::default()).expect("lm failed");

        let expected = (2.0_f64).sqrt();

        assert!(newton_result.converged);
        assert!((newton_result.x[0] - expected).abs() < 1e-5);

        assert!(broyden_result.converged);
        assert!((broyden_result.x[0] - expected).abs() < 1e-5);

        assert!(lm_result.converged);
        assert!((lm_result.x[0] - expected).abs() < 1e-4);
    }
}
