//! Nelder-Mead simplex algorithm for derivative-free minimization.

#![allow(clippy::needless_range_loop)]

use super::{MinimizeOptions, MultiMinimizeResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::ZERO_THRESHOLD;

/// Nelder-Mead simplex algorithm for derivative-free minimization.
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
/// Nelder-Mead is robust and doesn't require derivatives, but convergence
/// can be slow for high-dimensional problems (n > 10).
pub fn nelder_mead<F>(
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
            context: "nelder_mead: empty initial guess".to_string(),
        });
    }

    // Simplex parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    // Initialize simplex with n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    // Create remaining vertices by perturbing each dimension
    for i in 0..n {
        let mut vertex = x0.to_vec();
        let delta = if x0[i].abs() < ZERO_THRESHOLD {
            0.00025
        } else {
            0.05 * x0[i].abs()
        };
        vertex[i] += delta;
        simplex.push(vertex);
    }

    // Compute function values at all vertices
    let mut f_values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut nfev = n + 1;

    for iter in 0..options.max_iter {
        // Sort vertices by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

        // Reorder simplex and f_values
        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_f: Vec<f64> = indices.iter().map(|&i| f_values[i]).collect();
        simplex = sorted_simplex;
        f_values = sorted_f;

        // Check convergence
        let f_best = f_values[0];
        let f_worst = f_values[n];
        let f_range = (f_worst - f_best).abs();

        // Compute simplex diameter
        let mut max_dist = 0.0_f64;
        for i in 1..=n {
            let dist: f64 = simplex[0]
                .iter()
                .zip(simplex[i].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            max_dist = max_dist.max(dist);
        }

        if f_range < options.f_tol && max_dist < options.x_tol {
            return Ok(MultiMinimizeResult {
                x: simplex[0].clone(),
                fun: f_values[0],
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute centroid of all vertices except worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let mut x_r = vec![0.0; n];
        for j in 0..n {
            x_r[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j]);
        }
        let f_r = f(&x_r);
        nfev += 1;

        if f_r < f_values[0] {
            // Try expansion
            let mut x_e = vec![0.0; n];
            for j in 0..n {
                x_e[j] = centroid[j] + gamma * (x_r[j] - centroid[j]);
            }
            let f_e = f(&x_e);
            nfev += 1;

            if f_e < f_r {
                simplex[n] = x_e;
                f_values[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_values[n] = f_r;
            }
        } else if f_r < f_values[n - 1] {
            // Accept reflection
            simplex[n] = x_r;
            f_values[n] = f_r;
        } else {
            // Contraction
            let (x_c, f_c) = if f_r < f_values[n] {
                // Outside contraction
                let mut x_c = vec![0.0; n];
                for j in 0..n {
                    x_c[j] = centroid[j] + rho * (x_r[j] - centroid[j]);
                }
                let f_c = f(&x_c);
                nfev += 1;
                (x_c, f_c)
            } else {
                // Inside contraction
                let mut x_c = vec![0.0; n];
                for j in 0..n {
                    x_c[j] = centroid[j] - rho * (centroid[j] - simplex[n][j]);
                }
                let f_c = f(&x_c);
                nfev += 1;
                (x_c, f_c)
            };

            if f_c < f_values[n].min(f_r) {
                simplex[n] = x_c;
                f_values[n] = f_c;
            } else {
                // Shrink
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    f_values[i] = f(&simplex[i]);
                    nfev += 1;
                }
            }
        }
    }

    // Return best result even if not converged
    let mut indices: Vec<usize> = (0..=n).collect();
    indices.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

    Ok(MultiMinimizeResult {
        x: simplex[indices[0]].clone(),
        fun: f_values[indices[0]],
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
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
    fn test_nelder_mead_sphere() {
        let result = nelder_mead(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default())
            .expect("nelder_mead failed");

        assert!(result.converged);
        assert!(result.fun < 1e-8);
        for xi in &result.x {
            assert!(xi.abs() < 1e-4);
        }
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        let result = nelder_mead(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default())
            .expect("nelder_mead failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let mut opts = MinimizeOptions::default();
        opts.max_iter = 2000;

        let result = nelder_mead(rosenbrock, &[0.0, 0.0], &opts).expect("nelder_mead failed");

        // Rosenbrock is challenging, just check it gets close
        assert!((result.x[0] - 1.0).abs() < 0.1);
        assert!((result.x[1] - 1.0).abs() < 0.1);
    }
}
