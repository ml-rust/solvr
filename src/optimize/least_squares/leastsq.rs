//! Levenberg-Marquardt algorithm for unbounded nonlinear least squares.

#![allow(clippy::needless_range_loop)]

use super::{LeastSquaresOptions, LeastSquaresResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    ZERO_THRESHOLD, finite_difference_jacobian, norm, norm_squared, solve_linear_system,
};

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
    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;

    for iter in 0..options.max_iter {
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

        let neg_jtf: Vec<f64> = jtf.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jtj, &neg_jtf) {
            Some(dx) => dx,
            None => {
                lambda *= lambda_up;
                continue;
            }
        };

        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
        let fx_new = f(&x_new);
        nfev += 1;
        let cost_new = norm_squared(&fx_new);

        if cost_new < cost {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leastsq_linear_fit() {
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

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
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
        assert!(result.cost < 1e-8);
    }

    #[test]
    fn test_leastsq_exponential_fit() {
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
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_leastsq_quadratic_fit() {
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
        assert!((result.x[0] - 0.5).abs() < 1e-4);
        assert!((result.x[1] - (-2.0)).abs() < 1e-4);
        assert!((result.x[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_leastsq_noisy_data() {
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
        assert!((result.x[0] - 1.5).abs() < 0.1);
        assert!((result.x[1] - 0.8).abs() < 0.05);
    }

    #[test]
    fn test_empty_input() {
        let result = leastsq(|_: &[f64]| vec![], &[], &LeastSquaresOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }
}
