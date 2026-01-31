//! Conjugate gradient method for minimization.

#![allow(clippy::needless_range_loop)]

use super::{MinimizeOptions, MultiMinimizeResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{finite_difference_gradient_forward, norm};

/// Conjugate gradient method for minimization.
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
/// Uses Polak-Ribière conjugate gradient with restarts.
/// Requires less memory than BFGS (no Hessian approximation).
pub fn conjugate_gradient<F>(
    f: F,
    x0: &[f64],
    options: &MinimizeOptions,
) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "conjugate_gradient: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    let mut grad = finite_difference_gradient_forward(&f, &x, fx, options.eps);
    nfev += n;

    // Initial direction is negative gradient
    let mut p: Vec<f64> = grad.iter().map(|g| -g).collect();
    let mut grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();

    for iter in 0..options.max_iter {
        // Check gradient convergence
        if grad_norm_sq.sqrt() < options.g_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Line search
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

        // Polak-Ribière beta
        let grad_new_norm_sq: f64 = grad_new.iter().map(|g| g * g).sum();
        let grad_diff_dot_new: f64 = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(gn, g)| gn * (gn - g))
            .sum();

        let beta = (grad_diff_dot_new / grad_norm_sq).max(0.0);

        // Update direction: p = -grad_new + beta * p
        for i in 0..n {
            p[i] = -grad_new[i] + beta * p[i];
        }

        // Restart if direction is not descent
        let grad_dot_p: f64 = grad_new.iter().zip(p.iter()).map(|(g, d)| g * d).sum();
        if grad_dot_p >= 0.0 {
            for i in 0..n {
                p[i] = -grad_new[i];
            }
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
        grad_norm_sq = grad_new_norm_sq;
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

    #[test]
    fn test_cg_sphere() {
        let result = conjugate_gradient(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default())
            .expect("cg failed");

        assert!(result.converged);
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_cg_quadratic() {
        let result = conjugate_gradient(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default())
            .expect("cg failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_cg_empty_input() {
        let result = conjugate_gradient(sphere, &[], &MinimizeOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }
}
