//! BFGS quasi-Newton method for minimization.

#![allow(clippy::needless_range_loop)]

use super::{MinimizeOptions, MultiMinimizeResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{SINGULAR_THRESHOLD, finite_difference_gradient_forward, norm};

/// BFGS quasi-Newton method for minimization.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f`
///
/// # Note
/// BFGS uses gradient information and maintains an approximation of the
/// inverse Hessian. It has superlinear convergence near the minimum.
/// Uses finite differences for gradient approximation.
pub fn bfgs<F>(f: F, x0: &[f64], options: &MinimizeOptions) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "bfgs: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    let mut grad = finite_difference_gradient_forward(&f, &x, fx, options.eps);
    nfev += n;

    // Initialize inverse Hessian approximation to identity
    let mut h_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iter in 0..options.max_iter {
        // Check gradient convergence
        let grad_norm = norm(&grad);
        if grad_norm < options.g_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute search direction: p = -H_inv * grad
        let mut p = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p[i] -= h_inv[i][j] * grad[j];
            }
        }

        // Line search with Armijo condition
        let (_alpha, x_new, fx_new, evals) = backtracking_line_search(&f, &x, &p, fx, &grad);
        nfev += evals;

        // Check convergence
        let dx = norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>(),
        );
        if dx < options.x_tol || (fx - fx_new).abs() < options.f_tol {
            return Ok(MultiMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient_forward(&f, &x_new, fx_new, options.eps);
        nfev += n;

        // BFGS update
        // s = x_new - x
        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        // y = grad_new - grad
        let y: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(a, b)| a - b)
            .collect();

        // rho = 1 / (y^T * s)
        let ys: f64 = y.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
        if ys.abs() > SINGULAR_THRESHOLD {
            let rho = 1.0 / ys;

            // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
            // Simplified BFGS update
            let mut h_y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    h_y[i] += h_inv[i][j] * y[j];
                }
            }

            let yhy: f64 = y.iter().zip(h_y.iter()).map(|(a, b)| a * b).sum();

            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] += rho * (1.0 + rho * yhy) * s[i] * s[j]
                        - rho * (s[i] * h_y[j] + h_y[i] * s[j]);
                }
            }
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    Ok(MultiMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Backtracking line search with Armijo condition.
/// Returns (step_size, new_x, new_fx, num_evaluations).
fn backtracking_line_search<F>(
    f: &F,
    x: &[f64],
    p: &[f64],
    fx: f64,
    grad: &[f64],
) -> (f64, Vec<f64>, f64, usize)
where
    F: Fn(&[f64]) -> f64,
{
    let c = 0.0001; // Armijo constant
    let rho = 0.5; // Step reduction factor

    let grad_dot_p: f64 = grad.iter().zip(p.iter()).map(|(g, d)| g * d).sum();

    let mut alpha = 1.0;
    let mut nfev = 0;

    for _ in 0..50 {
        let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(a, d)| a + alpha * d).collect();
        let fx_new = f(&x_new);
        nfev += 1;

        // Armijo condition
        if fx_new <= fx + c * alpha * grad_dot_p {
            return (alpha, x_new, fx_new, nfev);
        }

        alpha *= rho;
    }

    // Return current point if line search fails
    (0.0, x.to_vec(), fx, nfev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn quadratic_2d(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    }

    #[test]
    fn test_bfgs_sphere() {
        let result =
            bfgs(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default()).expect("bfgs failed");

        assert!(result.converged);
        assert!(result.fun < 1e-8);
    }

    #[test]
    fn test_bfgs_quadratic() {
        let result =
            bfgs(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default()).expect("bfgs failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let mut opts = MinimizeOptions::default();
        opts.max_iter = 500;

        let result = bfgs(rosenbrock, &[0.0, 0.0], &opts).expect("bfgs failed");

        assert!((result.x[0] - 1.0).abs() < 0.01);
        assert!((result.x[1] - 1.0).abs() < 0.01);
    }
}
