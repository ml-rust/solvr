//! Least squares optimization and curve fitting.
//!
//! This module provides methods for solving nonlinear least squares problems:
//! minimize ||f(x)||^2 = sum(f_i(x)^2)
//!
//! where f: R^n -> R^m is a vector-valued function (residuals).

mod bounded;
mod leastsq;

pub use bounded::least_squares;
pub use leastsq::leastsq;

use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Options for least squares optimization.
#[derive(Debug, Clone)]
pub struct LeastSquaresOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (change in cost)
    pub f_tol: f64,
    /// Tolerance for convergence (change in parameters)
    pub x_tol: f64,
    /// Tolerance for convergence (gradient norm)
    pub g_tol: f64,
    /// Step size for finite difference Jacobian
    pub eps: f64,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            f_tol: 1e-8,
            x_tol: 1e-8,
            g_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result from least squares optimization.
#[derive(Debug, Clone)]
pub struct LeastSquaresResult {
    /// The optimal parameters found
    pub x: Vec<f64>,
    /// Residual vector at solution
    pub residuals: Vec<f64>,
    /// Sum of squared residuals (cost)
    pub cost: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}

/// Fit a model function to data using nonlinear least squares.
///
/// # Arguments
/// * `model` - Model function: f(x, params) -> y
/// * `x_data` - Independent variable data points
/// * `y_data` - Dependent variable data points
/// * `p0` - Initial parameter guess
/// * `options` - Solver options
///
/// # Returns
/// Optimal parameters fitting the model to data
///
/// # Example
/// ```ignore
/// // Fit y = a * exp(-b * x)
/// let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();
/// let result = curve_fit(model, &x_data, &y_data, &[1.0, 1.0], &options)?;
/// ```
pub fn curve_fit<F>(
    model: F,
    x_data: &[f64],
    y_data: &[f64],
    p0: &[f64],
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(f64, &[f64]) -> f64,
{
    if x_data.len() != y_data.len() {
        return Err(OptimizeError::InvalidInput {
            context: "curve_fit: x_data and y_data must have same length".to_string(),
        });
    }

    if x_data.is_empty() {
        return Err(OptimizeError::InvalidInput {
            context: "curve_fit: data arrays are empty".to_string(),
        });
    }

    let residual_fn = |params: &[f64]| -> Vec<f64> {
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| model(x, params) - y)
            .collect()
    };

    leastsq(residual_fn, p0, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_fit_exponential() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();

        let result = curve_fit(
            model,
            &x_data,
            &y_data,
            &[1.0, 1.0],
            &LeastSquaresOptions::default(),
        )
        .expect("curve_fit failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_curve_fit_gaussian() {
        let x_data: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.25).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 3.0 * (-(x - 1.0).powi(2) / (2.0 * 2.0 * 2.0)).exp())
            .collect();

        let model = |x: f64, p: &[f64]| p[0] * (-(x - p[1]).powi(2) / (2.0 * p[2] * p[2])).exp();

        let result = curve_fit(
            model,
            &x_data,
            &y_data,
            &[1.0, 0.0, 1.0],
            &LeastSquaresOptions::default(),
        )
        .expect("curve_fit failed");

        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 0.1);
        assert!((result.x[1] - 1.0).abs() < 0.1);
        assert!((result.x[2].abs() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_curve_fit_mismatched_data() {
        let result = curve_fit(
            |_, _| 0.0,
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0],
            &[1.0],
            &LeastSquaresOptions::default(),
        );
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }
}
