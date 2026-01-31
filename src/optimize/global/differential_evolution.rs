//! Differential Evolution global optimizer.

use super::{GlobalOptions, GlobalResult, SimpleRng};
use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Differential Evolution global optimizer.
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
/// DE is a population-based stochastic optimizer that works well for
/// non-smooth, non-convex functions. It doesn't require gradients.
pub fn differential_evolution<F>(
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
            context: "differential_evolution: empty bounds".to_string(),
        });
    }

    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInterval {
                a: lo,
                b: hi,
                context: format!("differential_evolution: invalid bounds for dimension {}", i),
            });
        }
    }

    // DE parameters
    let pop_size = (15 * n).max(25);
    let f_scale = 0.8;
    let cr = 0.9;

    let seed = options.seed.unwrap_or(12345);
    let mut rng = SimpleRng::new(seed);

    // Initialize population uniformly within bounds
    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.next_range(lo, hi))
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population.iter().map(|x| f(x)).collect();
    let mut nfev = pop_size;

    let mut best_idx = 0;
    let mut best_fitness = fitness[0];
    for (i, &fit) in fitness.iter().enumerate() {
        if fit < best_fitness {
            best_fitness = fit;
            best_idx = i;
        }
    }

    for iter in 0..options.max_iter {
        let fitness_range = fitness.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        if fitness_range < options.tol {
            return Ok(GlobalResult {
                x: population[best_idx].clone(),
                fun: best_fitness,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        for i in 0..pop_size {
            // Select three distinct random individuals (not i)
            let mut r = [0usize; 3];
            for j in 0..3 {
                loop {
                    r[j] = rng.next_int(pop_size);
                    if r[j] != i && (j == 0 || (r[j] != r[0] && (j == 1 || r[j] != r[1]))) {
                        break;
                    }
                }
            }

            // Create mutant vector
            let mut mutant: Vec<f64> = (0..n)
                .map(|j| {
                    population[r[0]][j] + f_scale * (population[r[1]][j] - population[r[2]][j])
                })
                .collect();

            for j in 0..n {
                mutant[j] = mutant[j].clamp(bounds[j].0, bounds[j].1);
            }

            // Crossover
            let j_rand = rng.next_int(n);
            let trial: Vec<f64> = (0..n)
                .map(|j| {
                    if rng.next_f64() < cr || j == j_rand {
                        mutant[j]
                    } else {
                        population[i][j]
                    }
                })
                .collect();

            // Selection
            let trial_fitness = f(&trial);
            nfev += 1;

            if trial_fitness <= fitness[i] {
                population[i] = trial;
                fitness[i] = trial_fitness;

                if trial_fitness < best_fitness {
                    best_fitness = trial_fitness;
                    best_idx = i;
                }
            }
        }
    }

    Ok(GlobalResult {
        x: population[best_idx].clone(),
        fun: best_fitness,
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

    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    #[test]
    fn test_de_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            seed: Some(42),
            ..Default::default()
        };

        let result = differential_evolution(sphere, &bounds, &opts).expect("DE failed");
        assert!(result.fun < 1e-6);
        for xi in &result.x {
            assert!(xi.abs() < 0.01);
        }
    }

    #[test]
    fn test_de_rastrigin() {
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = differential_evolution(rastrigin, &bounds, &opts).expect("DE failed");
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_de_rosenbrock() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = differential_evolution(rosenbrock, &bounds, &opts).expect("DE failed");
        assert!(result.fun < 0.1);
    }

    #[test]
    fn test_de_empty_bounds() {
        let result = differential_evolution(sphere, &[], &GlobalOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_de_invalid_bounds() {
        let bounds = vec![(5.0, -5.0)];
        let result = differential_evolution(sphere, &bounds, &GlobalOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInterval { .. })
        ));
    }
}
