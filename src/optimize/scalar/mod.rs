//! Scalar (1D) root finding and minimization algorithms.
//!
//! This module provides methods for finding roots and minima of univariate functions.
//! Root finding methods require a function `f: (f64) -> f64`.
//! Most methods are bracketing (require an interval [a, b] where f(a)*f(b) < 0).

mod minimization;
mod root_finding;

pub use minimization::{minimize_scalar_bounded, minimize_scalar_brent, minimize_scalar_golden};
pub use root_finding::{bisect, brentq, newton, ridder, secant};

/// Options for scalar root finding and minimization.
#[derive(Debug, Clone)]
pub struct ScalarOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Absolute tolerance for convergence (root value)
    pub tol: f64,
    /// Relative tolerance for convergence (width of interval)
    pub rtol: f64,
}

impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-12,
            rtol: 1e-12,
        }
    }
}

/// Result from a root finding method.
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The root found
    pub root: f64,
    /// Function value at root
    pub function_value: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final bracket width (for bracketing methods)
    pub bracket_width: f64,
}

/// Result from a minimization method.
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    /// The minimum point found
    pub x: f64,
    /// Function value at minimum
    pub f_min: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final bracket width (for bracketing methods)
    pub bracket_width: f64,
}
