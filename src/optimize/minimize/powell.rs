//! Powell's direction set method for derivative-free minimization.

#![allow(clippy::needless_range_loop)]

use super::{MinimizeOptions, MultiMinimizeResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{SINGULAR_THRESHOLD, ZERO_THRESHOLD, norm};

/// Powell's direction set method for derivative-free minimization.
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
/// Powell's method performs successive line searches along conjugate directions.
/// It doesn't require derivatives but can be faster than Nelder-Mead.
pub fn powell<F>(f: F, x0: &[f64], options: &MinimizeOptions) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "powell: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    // Initialize direction set to coordinate directions
    let mut directions: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut d = vec![0.0; n];
            d[i] = 1.0;
            d
        })
        .collect();

    for iter in 0..options.max_iter {
        let x_start = x.clone();
        let fx_start = fx;

        // Perform line search along each direction
        let mut max_decrease = 0.0;
        let mut max_decrease_idx = 0;

        for (i, dir) in directions.iter().enumerate() {
            let (x_new, fx_new, evals) = line_search_quadratic(&f, &x, dir, fx);
            nfev += evals;

            let decrease = fx - fx_new;
            if decrease > max_decrease {
                max_decrease = decrease;
                max_decrease_idx = i;
            }

            x = x_new;
            fx = fx_new;
        }

        // Check convergence
        let dx: f64 = x
            .iter()
            .zip(x_start.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        if dx < options.x_tol || (fx_start - fx).abs() < options.f_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new direction
        let mut new_dir: Vec<f64> = x.iter().zip(x_start.iter()).map(|(a, b)| a - b).collect();
        let dir_norm = norm(&new_dir);
        if dir_norm > SINGULAR_THRESHOLD {
            for d in &mut new_dir {
                *d /= dir_norm;
            }

            // Line search along new direction
            let (x_new, fx_new, evals) = line_search_quadratic(&f, &x, &new_dir, fx);
            nfev += evals;

            // Replace direction with largest decrease
            if fx_new < fx {
                directions[max_decrease_idx] = new_dir;
                x = x_new;
                fx = fx_new;
            }
        }
    }

    Ok(MultiMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Simple quadratic line search.
/// Returns (new_x, new_fx, num_evaluations).
fn line_search_quadratic<F>(f: &F, x: &[f64], direction: &[f64], fx: f64) -> (Vec<f64>, f64, usize)
where
    F: Fn(&[f64]) -> f64,
{
    let mut nfev = 0;

    // Bracket the minimum
    let f_alpha = fx;

    let mut beta = 1.0;
    let x_beta: Vec<f64> = x
        .iter()
        .zip(direction.iter())
        .map(|(a, d)| a + beta * d)
        .collect();
    let mut f_beta = f(&x_beta);
    nfev += 1;

    // If f_beta > f_alpha, reduce step size
    while f_beta > f_alpha && beta > ZERO_THRESHOLD {
        beta *= 0.5;
        let x_beta: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(a, d)| a + beta * d)
            .collect();
        f_beta = f(&x_beta);
        nfev += 1;
    }

    // If still not better, try negative direction
    if f_beta >= f_alpha {
        beta = -1.0;
        let x_beta: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(a, d)| a + beta * d)
            .collect();
        f_beta = f(&x_beta);
        nfev += 1;

        while f_beta > f_alpha && beta.abs() > ZERO_THRESHOLD {
            beta *= 0.5;
            let x_beta: Vec<f64> = x
                .iter()
                .zip(direction.iter())
                .map(|(a, d)| a + beta * d)
                .collect();
            f_beta = f(&x_beta);
            nfev += 1;
        }
    }

    if f_beta >= f_alpha {
        return (x.to_vec(), fx, nfev);
    }

    // Golden section search to refine
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0;
    let inv_phi2 = 1.0 - inv_phi;

    let (mut a, mut b) = if beta > 0.0 { (0.0, beta) } else { (beta, 0.0) };

    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);

    let x_at = |t: f64| -> Vec<f64> {
        x.iter()
            .zip(direction.iter())
            .map(|(a, d)| a + t * d)
            .collect()
    };

    let mut f1 = f(&x_at(x1));
    let mut f2 = f(&x_at(x2));
    nfev += 2;

    for _ in 0..20 {
        if (b - a).abs() < 1e-8 {
            break;
        }

        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(&x_at(x1));
            nfev += 1;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(&x_at(x2));
            nfev += 1;
        }
    }

    let best_alpha = 0.5 * (a + b);
    let x_best = x_at(best_alpha);
    let f_best = f(&x_best);
    nfev += 1;

    (x_best, f_best, nfev)
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
    fn test_powell_sphere() {
        let result =
            powell(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default()).expect("powell failed");

        assert!(result.converged);
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_powell_quadratic() {
        let result =
            powell(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default()).expect("powell failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }
}
