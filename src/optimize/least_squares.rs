//! Least squares optimization and curve fitting.
//!
//! This module provides methods for solving nonlinear least squares problems:
//! minimize ||f(x)||^2 = sum(f_i(x)^2)
//!
//! where f: R^n -> R^m is a vector-valued function (residuals).

// Indexed loops are clearer for matrix operations
#![allow(clippy::needless_range_loop)]

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    ZERO_THRESHOLD, finite_difference_jacobian, norm, norm_squared, solve_linear_system,
};

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

/// Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// # Arguments
/// * `f` - Residual function f: R^n -> R^m
/// * `x0` - Initial parameter guess
/// * `options` - Solver options
///
/// # Returns
/// Optimal parameters minimizing ||f(x)||^2
///
/// # Note
/// LM interpolates between Gauss-Newton and gradient descent.
/// It's the most popular algorithm for nonlinear least squares.
pub fn leastsq<F>(
    f: F,
    x0: &[f64],
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "leastsq: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let m = fx.len();
    let mut nfev = 1;

    if m == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "leastsq: residual function returns empty vector".to_string(),
        });
    }

    let mut cost = norm_squared(&fx);
    let mut lambda = 0.001; // Damping parameter
    let lambda_up = 10.0;
    let lambda_down = 0.1;

    for iter in 0..options.max_iter {
        // Check convergence
        if cost < options.f_tol {
            return Ok(LeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute Jacobian
        let jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);
        nfev += n;

        // Compute J^T J + lambda * diag(J^T J)
        let mut jtj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    jtj[i][j] += jacobian[k][i] * jacobian[k][j];
                }
            }
        }

        // Add damping to diagonal
        for i in 0..n {
            jtj[i][i] *= 1.0 + lambda;
            if jtj[i][i] < ZERO_THRESHOLD {
                jtj[i][i] = ZERO_THRESHOLD; // Ensure positive definite
            }
        }

        // Compute J^T f
        let mut jtf = vec![0.0; n];
        for i in 0..n {
            for k in 0..m {
                jtf[i] += jacobian[k][i] * fx[k];
            }
        }

        // Check gradient convergence
        let grad_norm = norm(&jtf);
        if grad_norm < options.g_tol {
            return Ok(LeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Solve (J^T J + lambda * D) dx = -J^T f
        let neg_jtf: Vec<f64> = jtf.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jtj, &neg_jtf) {
            Some(dx) => dx,
            None => {
                lambda *= lambda_up;
                continue;
            }
        };

        // Try the step
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
        let fx_new = f(&x_new);
        nfev += 1;
        let cost_new = norm_squared(&fx_new);

        // Accept or reject step
        if cost_new < cost {
            // Check x convergence
            let dx_norm = norm(&dx);
            if dx_norm < options.x_tol {
                return Ok(LeastSquaresResult {
                    x: x_new,
                    residuals: fx_new,
                    cost: cost_new,
                    iterations: iter + 1,
                    nfev,
                    converged: true,
                });
            }

            x = x_new;
            fx = fx_new;
            cost = cost_new;
            lambda *= lambda_down;
        } else {
            lambda *= lambda_up;
        }

        // Prevent lambda from becoming too large or too small
        lambda = lambda.clamp(ZERO_THRESHOLD, 1e10);
    }

    Ok(LeastSquaresResult {
        x,
        residuals: fx,
        cost,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Least squares with optional bounds using Levenberg-Marquardt.
///
/// # Arguments
/// * `f` - Residual function f: R^n -> R^m
/// * `x0` - Initial parameter guess
/// * `bounds` - Optional bounds (lower, upper) for each parameter
/// * `options` - Solver options
///
/// # Returns
/// Optimal parameters minimizing ||f(x)||^2 within bounds
///
/// # Note
/// For unbounded problems, this uses the standard LM algorithm.
/// For bounded problems, it projects steps onto the feasible region.
pub fn least_squares<F>(
    f: F,
    x0: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: empty initial guess".to_string(),
        });
    }

    // If no bounds, just use leastsq
    if bounds.is_none() {
        return leastsq(f, x0, options);
    }

    // Validate bounds
    let (lower, upper) = bounds.unwrap();
    if lower.len() != n || upper.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: bounds dimension mismatch".to_string(),
        });
    }

    let lower = lower.to_vec();
    let upper = upper.to_vec();

    // Project initial guess onto bounds
    let mut x: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.clamp(lower[i], upper[i]))
        .collect();

    let mut fx = f(&x);
    let m = fx.len();
    let mut nfev = 1;

    if m == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: residual function returns empty vector".to_string(),
        });
    }

    let mut cost = norm_squared(&fx);
    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;

    for iter in 0..options.max_iter {
        // Check convergence
        if cost < options.f_tol {
            return Ok(LeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute Jacobian
        let jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);
        nfev += n;

        // Compute J^T J + lambda * diag(J^T J)
        let mut jtj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    jtj[i][j] += jacobian[k][i] * jacobian[k][j];
                }
            }
        }

        // Add damping to diagonal
        for i in 0..n {
            jtj[i][i] *= 1.0 + lambda;
            if jtj[i][i] < ZERO_THRESHOLD {
                jtj[i][i] = ZERO_THRESHOLD;
            }
        }

        // Compute J^T f
        let mut jtf = vec![0.0; n];
        for i in 0..n {
            for k in 0..m {
                jtf[i] += jacobian[k][i] * fx[k];
            }
        }

        // Check gradient convergence
        let grad_norm = norm(&jtf);
        if grad_norm < options.g_tol {
            return Ok(LeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Solve (J^T J + lambda * D) dx = -J^T f
        let neg_jtf: Vec<f64> = jtf.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jtj, &neg_jtf) {
            Some(dx) => dx,
            None => {
                lambda *= lambda_up;
                continue;
            }
        };

        // Project step onto bounds
        let x_new: Vec<f64> = x
            .iter()
            .zip(dx.iter())
            .enumerate()
            .map(|(i, (&xi, &di))| (xi + di).clamp(lower[i], upper[i]))
            .collect();

        let fx_new = f(&x_new);
        nfev += 1;
        let cost_new = norm_squared(&fx_new);

        // Accept or reject step
        if cost_new < cost {
            let dx_norm = norm(
                &x_new
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a - b)
                    .collect::<Vec<f64>>(),
            );
            if dx_norm < options.x_tol {
                return Ok(LeastSquaresResult {
                    x: x_new,
                    residuals: fx_new,
                    cost: cost_new,
                    iterations: iter + 1,
                    nfev,
                    converged: true,
                });
            }

            x = x_new;
            fx = fx_new;
            cost = cost_new;
            lambda *= lambda_down;
        } else {
            lambda *= lambda_up;
        }

        lambda = lambda.clamp(ZERO_THRESHOLD, 1e10);
    }

    Ok(LeastSquaresResult {
        x,
        residuals: fx,
        cost,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
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

    // Create residual function: f_i(p) = model(x_i, p) - y_i
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
    fn test_leastsq_linear_fit() {
        // Fit y = a + b*x to data points
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 1 + 2*x

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        let result = leastsq(residual, &[0.0, 0.0], &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4); // a = 1
        assert!((result.x[1] - 2.0).abs() < 1e-4); // b = 2
        assert!(result.cost < 1e-8);
    }

    #[test]
    fn test_leastsq_exponential_fit() {
        // Fit y = a * exp(-b * x)
        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] * (-p[1] * x).exp() - y)
                .collect()
        };

        let result = leastsq(residual, &[1.0, 1.0], &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4); // a = 2
        assert!((result.x[1] - 0.5).abs() < 1e-4); // b = 0.5
    }

    #[test]
    fn test_leastsq_quadratic_fit() {
        // Fit y = a*x^2 + b*x + c
        let x_data: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 0.5 * x * x - 2.0 * x + 1.0)
            .collect();

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] * x * x + p[1] * x + p[2] - y)
                .collect()
        };

        let result = leastsq(residual, &[0.0, 0.0, 0.0], &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        assert!((result.x[0] - 0.5).abs() < 1e-4); // a = 0.5
        assert!((result.x[1] - (-2.0)).abs() < 1e-4); // b = -2
        assert!((result.x[2] - 1.0).abs() < 1e-4); // c = 1
    }

    #[test]
    fn test_least_squares_unbounded() {
        // Same as leastsq test - uses LM when no bounds
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        let result = least_squares(residual, &[0.0, 0.0], None, &LeastSquaresOptions::default())
            .expect("least_squares failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_least_squares_bounded() {
        // Fit with bounds on parameters
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // True params: a=1, b=2

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        // Bound b to [0, 1.5] - should get b at or near 1.5 (constraint active)
        let lower = vec![-10.0, 0.0];
        let upper = vec![10.0, 1.5];

        let result = least_squares(
            residual,
            &[1.0, 1.0], // Start closer to solution
            Some((&lower, &upper)),
            &LeastSquaresOptions::default(),
        )
        .expect("least_squares failed");

        eprintln!("bounded result: x = {:?}, cost = {}", result.x, result.cost);

        // b should respect the bound
        assert!(result.x[1] <= 1.5 + 1e-6);
        assert!(result.x[1] >= 0.0 - 1e-6);
    }

    #[test]
    fn test_curve_fit_exponential() {
        // Fit y = a * exp(-b * x)
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
        // Fit y = a * exp(-(x-mu)^2 / (2*sigma^2))
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
        assert!((result.x[0] - 3.0).abs() < 0.1); // amplitude
        assert!((result.x[1] - 1.0).abs() < 0.1); // mean
        assert!((result.x[2].abs() - 2.0).abs() < 0.1); // sigma (could be negative)
    }

    #[test]
    fn test_leastsq_noisy_data() {
        // Fit with some noise
        let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let noise = vec![
            0.1, -0.05, 0.02, -0.08, 0.03, 0.07, -0.02, 0.04, -0.06, 0.01, -0.03, 0.05, -0.01,
            0.06, -0.04, 0.02, -0.07, 0.03, -0.02, 0.05,
        ];
        let y_data: Vec<f64> = x_data
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| 1.5 + 0.8 * x + n)
            .collect();

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        let result = leastsq(residual, &[0.0, 0.0], &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        // Should be close to true values despite noise
        assert!((result.x[0] - 1.5).abs() < 0.1);
        assert!((result.x[1] - 0.8).abs() < 0.05);
    }

    #[test]
    fn test_empty_input() {
        let result = leastsq(|_: &[f64]| vec![], &[], &LeastSquaresOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
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
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }
}
