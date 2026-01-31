//! Quasi-Newton methods for multivariate root finding.

#![allow(clippy::needless_range_loop)]

use super::{MultiRootResult, RootOptions};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    SINGULAR_THRESHOLD, ZERO_THRESHOLD, finite_difference_jacobian, norm, solve_linear_system,
};

/// Broyden's method (rank-1 update) for systems of nonlinear equations.
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
/// Broyden's method is a quasi-Newton method that approximates the Jacobian
/// using rank-1 updates. It requires fewer function evaluations than Newton's
/// method but may converge more slowly.
pub fn broyden1<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "broyden1: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "broyden1: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    let mut jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);

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

        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jacobian, &neg_fx) {
            Some(dx) => dx,
            None => {
                jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);
                match solve_linear_system(&jacobian, &neg_fx) {
                    Some(dx) => dx,
                    None => {
                        return Err(OptimizeError::NumericalError {
                            message: "Singular Jacobian in broyden1".to_string(),
                        });
                    }
                }
            }
        };

        let _x_old = x.clone();
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

        let fx_new = f(&x);

        // Broyden rank-1 update
        let df: Vec<f64> = fx_new.iter().zip(fx.iter()).map(|(a, b)| a - b).collect();

        let j_dx: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| jacobian[i][j] * dx[j]).sum())
            .collect();

        let diff: Vec<f64> = df.iter().zip(j_dx.iter()).map(|(a, b)| a - b).collect();

        let dx_dot_dx: f64 = dx.iter().map(|v| v * v).sum();

        if dx_dot_dx > SINGULAR_THRESHOLD {
            for i in 0..n {
                for j in 0..n {
                    jacobian[i][j] += diff[i] * dx[j] / dx_dot_dx;
                }
            }
        }

        fx = fx_new;
    }

    Ok(MultiRootResult {
        x,
        fun: fx.clone(),
        iterations: options.max_iter,
        residual_norm: norm(&fx),
        converged: false,
    })
}

/// Levenberg-Marquardt algorithm for systems of nonlinear equations.
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
/// Levenberg-Marquardt is a damped Newton method that interpolates between
/// Newton's method and gradient descent. It's more robust than Newton's method
/// for problems where the initial guess is far from the solution.
pub fn levenberg_marquardt<F>(
    f: F,
    x0: &[f64],
    options: &RootOptions,
) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "levenberg_marquardt: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "levenberg_marquardt: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;

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

        // Compute J^T * J + lambda * I
        let mut jtj: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    jtj[i][j] += jacobian[k][i] * jacobian[k][j];
                }
                if i == j {
                    jtj[i][j] += lambda;
                }
            }
        }

        let jtf: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|k| jacobian[k][i] * fx[k]).sum::<f64>())
            .collect();

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
        let new_res_norm = norm(&fx_new);

        if new_res_norm < res_norm {
            x = x_new;
            fx = fx_new;
            lambda *= lambda_down;

            if norm(&dx) < options.x_tol {
                return Ok(MultiRootResult {
                    x,
                    fun: fx.clone(),
                    iterations: iter + 1,
                    residual_norm: norm(&fx),
                    converged: true,
                });
            }
        } else {
            lambda *= lambda_up;
        }

        lambda = lambda.clamp(ZERO_THRESHOLD, 1e10);
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
    fn test_broyden1_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = broyden1(f, &[0.0, 0.0], &RootOptions::default()).expect("broyden1 failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_broyden1_nonlinear() {
        let f = |x: &[f64]| vec![x[0] * x[0] - x[1], x[0] + x[1] - 2.0];

        let result = broyden1(f, &[0.5, 0.5], &RootOptions::default()).expect("broyden1 failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_broyden1_dimension_mismatch() {
        let f = |x: &[f64]| vec![x[0], x[1], 0.0];

        let result = broyden1(f, &[1.0, 1.0], &RootOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_levenberg_marquardt_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = levenberg_marquardt(f, &[0.0, 0.0], &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_levenberg_marquardt_nonlinear() {
        let f = |x: &[f64]| vec![x[0] * x[0] * x[0] - x[1], x[0] + x[1] * x[1] * x[1] - 2.0];

        let result = levenberg_marquardt(f, &[0.5, 0.5], &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 1.0).abs() < 1e-4);
    }
}
