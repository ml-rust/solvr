//! Simulated annealing global optimizer.

use super::{GlobalOptions, GlobalResult, SimpleRng};
use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Simulated annealing global optimizer.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `bounds` - Bounds for each dimension: [(min, max), ...]
/// * `options` - Solver options
///
/// # Returns
/// Global minimum of `f` within bounds
///
/// # Note
/// Simulated annealing is inspired by the annealing process in metallurgy.
/// It gradually reduces "temperature" to settle into a global minimum.
pub fn simulated_annealing<F>(
    f: F,
    bounds: &[(f64, f64)],
    options: &GlobalOptions,
) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = bounds.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "simulated_annealing: empty bounds".to_string(),
        });
    }

    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInterval {
                a: lo,
                b: hi,
                context: format!("simulated_annealing: invalid bounds for dimension {}", i),
            });
        }
    }

    let seed = options.seed.unwrap_or(12345);
    let mut rng = SimpleRng::new(seed);

    // Parameters
    let t_initial: f64 = 5230.0;
    let t_final: f64 = 0.0001;
    let cooling_rate = (t_final / t_initial).powf(1.0 / options.max_iter as f64);

    // Initialize at random point within bounds
    let mut x_current: Vec<f64> = bounds
        .iter()
        .map(|&(lo, hi)| rng.next_range(lo, hi))
        .collect();
    let mut f_current = f(&x_current);
    let mut nfev = 1;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut temperature = t_initial;

    for iter in 0..options.max_iter {
        // Generate neighbor by perturbing current solution
        let scale = temperature / t_initial;
        let x_neighbor: Vec<f64> = x_current
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let range = bounds[i].1 - bounds[i].0;
                let delta = scale * range * (2.0 * rng.next_f64() - 1.0);
                (xi + delta).clamp(bounds[i].0, bounds[i].1)
            })
            .collect();

        let f_neighbor = f(&x_neighbor);
        nfev += 1;

        // Acceptance criterion
        let delta = f_neighbor - f_current;
        let accept = if delta < 0.0 {
            true
        } else {
            rng.next_f64() < (-delta / temperature).exp()
        };

        if accept {
            x_current = x_neighbor;
            f_current = f_neighbor;

            if f_current < f_best {
                x_best = x_current.clone();
                f_best = f_current;
            }
        }

        temperature *= cooling_rate;

        if temperature < t_final {
            return Ok(GlobalResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    Ok(GlobalResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    #[test]
    fn test_simulated_annealing_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 10000,
            seed: Some(42),
            ..Default::default()
        };

        let result = simulated_annealing(sphere, &bounds, &opts).expect("SA failed");
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_simulated_annealing_rastrigin() {
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let opts = GlobalOptions {
            max_iter: 10000,
            seed: Some(42),
            ..Default::default()
        };

        let result = simulated_annealing(rastrigin, &bounds, &opts).expect("SA failed");
        assert!(result.fun < 5.0);
    }
}
