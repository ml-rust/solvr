//! Bounded Levenberg-Marquardt algorithm for nonlinear least squares.

#![allow(clippy::needless_range_loop)]

use super::{LeastSquaresOptions, LeastSquaresResult, leastsq};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    ZERO_THRESHOLD, finite_difference_jacobian, norm, norm_squared, solve_linear_system,
};

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

    if bounds.is_none() {
        return leastsq(f, x0, options);
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_least_squares_unbounded() {
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
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        // Bound b to [0, 1.5]
        let lower = vec![-10.0, 0.0];
        let upper = vec![10.0, 1.5];

        let result = least_squares(
            residual,
            &[1.0, 1.0],
            Some((&lower, &upper)),
            &LeastSquaresOptions::default(),
        )
        .expect("least_squares failed");

        assert!(result.x[1] <= 1.5 + 1e-6);
        assert!(result.x[1] >= 0.0 - 1e-6);
    }
}
