//! Global optimization algorithms.
//!
//! This module provides methods for finding global minima of functions,
//! avoiding local minima traps that affect local optimization methods.

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::{MinimizeOptions, nelder_mead};

/// Options for global optimization.
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    /// Maximum number of iterations/generations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            seed: None,
        }
    }
}

/// Result from a global optimization method.
#[derive(Debug, Clone)]
pub struct GlobalResult {
    /// The global minimum point found
    pub x: Vec<f64>,
    /// Function value at minimum
    pub fun: f64,
    /// Number of iterations/generations
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}

/// Simple linear congruential generator for reproducible randomness.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    fn next_int(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

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

    // Validate bounds
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
    let pop_size = (15 * n).max(25); // Population size
    let f_scale = 0.8; // Differential weight (mutation factor)
    let cr = 0.9; // Crossover probability

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

    // Evaluate fitness
    let mut fitness: Vec<f64> = population.iter().map(|x| f(x)).collect();
    let mut nfev = pop_size;

    // Find best individual
    let mut best_idx = 0;
    let mut best_fitness = fitness[0];
    for (i, &fit) in fitness.iter().enumerate() {
        if fit < best_fitness {
            best_fitness = fit;
            best_idx = i;
        }
    }

    for iter in 0..options.max_iter {
        // Check convergence
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

        // Evolve each individual
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

            // Create mutant vector: v = x_r0 + F * (x_r1 - x_r2)
            let mut mutant: Vec<f64> = (0..n)
                .map(|j| {
                    population[r[0]][j] + f_scale * (population[r[1]][j] - population[r[2]][j])
                })
                .collect();

            // Clip to bounds
            for j in 0..n {
                mutant[j] = mutant[j].clamp(bounds[j].0, bounds[j].1);
            }

            // Crossover: create trial vector
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

/// Basin-hopping global optimizer.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Global minimum of `f`
///
/// # Note
/// Basin-hopping combines local minimization with random perturbations
/// to escape local minima. It's effective for functions with many local minima.
pub fn basinhopping<F>(f: F, x0: &[f64], options: &GlobalOptions) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "basinhopping: empty initial guess".to_string(),
        });
    }

    let seed = options.seed.unwrap_or(12345);
    let mut rng = SimpleRng::new(seed);

    // Parameters
    let step_size = 0.5;
    let temperature = 1.0;

    // Local minimization options
    let local_opts = MinimizeOptions {
        max_iter: 100,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    // Initial local minimization
    let local_result = nelder_mead(&f, x0, &local_opts)?;
    let mut x_current = local_result.x;
    let mut f_current = local_result.fun;
    let mut nfev = local_result.nfev;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;

    for iter in 0..options.max_iter {
        // Random perturbation
        let x_perturbed: Vec<f64> = x_current
            .iter()
            .map(|&xi| xi + step_size * (2.0 * rng.next_f64() - 1.0))
            .collect();

        // Local minimization from perturbed point
        let local_result = nelder_mead(&f, &x_perturbed, &local_opts)?;
        let x_new = local_result.x;
        let f_new = local_result.fun;
        nfev += local_result.nfev;

        // Metropolis acceptance criterion
        let delta_f = f_new - f_current;
        let accept = if delta_f < 0.0 {
            true
        } else {
            rng.next_f64() < (-delta_f / temperature).exp()
        };

        if accept {
            x_current = x_new;
            f_current = f_new;

            if f_current < f_best {
                x_best = x_current.clone();
                f_best = f_current;
            }
        }

        // Check convergence
        if (f_current - f_best).abs() < options.tol && iter > 10 {
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

    // Validate bounds
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
    let t_initial: f64 = 5230.0; // Initial temperature
    let t_final: f64 = 0.0001; // Final temperature
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
        let scale = temperature / t_initial; // Decreasing step size
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

        // Cool down
        temperature *= cooling_rate;

        // Check convergence
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

/// Dual annealing global optimizer.
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
/// Dual annealing combines simulated annealing with local search.
/// It's more efficient than pure simulated annealing for smooth functions.
pub fn dual_annealing<F>(
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
            context: "dual_annealing: empty bounds".to_string(),
        });
    }

    // Validate bounds
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInterval {
                a: lo,
                b: hi,
                context: format!("dual_annealing: invalid bounds for dimension {}", i),
            });
        }
    }

    let seed = options.seed.unwrap_or(12345);
    let mut rng = SimpleRng::new(seed);

    // Local minimization options
    let local_opts = MinimizeOptions {
        max_iter: 50,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    // Parameters
    let t_initial: f64 = 5230.0;
    let t_final: f64 = 0.001;
    let cooling_rate = (t_final / t_initial).powf(1.0 / (options.max_iter as f64 / 2.0));
    let local_search_interval = 10; // Do local search every N iterations

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
        // Occasionally do local search
        if iter > 0 && iter % local_search_interval == 0 {
            let local_result = nelder_mead(&f, &x_current, &local_opts)?;
            nfev += local_result.nfev;

            // Project back onto bounds
            let x_local: Vec<f64> = local_result
                .x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi.clamp(bounds[i].0, bounds[i].1))
                .collect();
            let f_local = f(&x_local);
            nfev += 1;

            if f_local < f_current {
                x_current = x_local;
                f_current = f_local;

                if f_current < f_best {
                    x_best = x_current.clone();
                    f_best = f_current;
                }
            }
        }

        // Generalized simulated annealing step (visiting distribution)
        let scale = temperature / t_initial;
        let x_neighbor: Vec<f64> = x_current
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let range = bounds[i].1 - bounds[i].0;
                // Use Cauchy distribution for visiting (heavier tails)
                let u = rng.next_f64();
                let cauchy = scale * range * (std::f64::consts::PI * (u - 0.5)).tan();
                (xi + cauchy * 0.1).clamp(bounds[i].0, bounds[i].1)
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

        // Cool down
        temperature *= cooling_rate;

        // Check convergence
        if temperature < t_final || (f_best < options.tol && iter > 100) {
            // Final local search
            let local_result = nelder_mead(&f, &x_best, &local_opts)?;
            nfev += local_result.nfev;

            let x_final: Vec<f64> = local_result
                .x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi.clamp(bounds[i].0, bounds[i].1))
                .collect();
            let f_final = f(&x_final);
            nfev += 1;

            if f_final < f_best {
                x_best = x_final;
                f_best = f_final;
            }

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

    // Rastrigin function: many local minima, global minimum at origin
    fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // Ackley function: many local minima
    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    // Rosenbrock function (banana function)
    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    // Simple sphere function
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

        // Should find global minimum near 0
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

        // Should find minimum near (1, 1)
        assert!(result.fun < 0.1);
    }

    #[test]
    fn test_basinhopping_sphere() {
        let opts = GlobalOptions {
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = basinhopping(sphere, &[2.0, 2.0, 2.0], &opts).expect("basinhopping failed");

        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_basinhopping_rosenbrock() {
        let opts = GlobalOptions {
            max_iter: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = basinhopping(rosenbrock, &[0.0, 0.0], &opts).expect("basinhopping failed");

        // Should find minimum near (1, 1)
        assert!(result.fun < 0.1);
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

        // SA is stochastic; on sphere should get reasonably close to minimum
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

        // SA might not find global minimum, but should get reasonably close
        assert!(result.fun < 5.0);
    }

    #[test]
    fn test_dual_annealing_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = dual_annealing(sphere, &bounds, &opts).expect("DA failed");

        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_dual_annealing_ackley() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let result = dual_annealing(ackley, &bounds, &opts).expect("DA failed");

        // Ackley global minimum is 0 at origin; dual annealing should get close
        assert!(result.fun < 3.0);
    }

    #[test]
    fn test_de_empty_bounds() {
        let result = differential_evolution(sphere, &[], &GlobalOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_de_invalid_bounds() {
        let bounds = vec![(5.0, -5.0)]; // Invalid: lo > hi
        let result = differential_evolution(sphere, &bounds, &GlobalOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_basinhopping_empty_input() {
        let result = basinhopping(sphere, &[], &GlobalOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_compare_methods() {
        // All methods should find a good solution for a simple problem
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let de_result = differential_evolution(&sphere, &bounds, &opts).expect("DE failed");
        let sa_result = simulated_annealing(&sphere, &bounds, &opts).expect("SA failed");
        let da_result = dual_annealing(&sphere, &bounds, &opts).expect("DA failed");

        // All should find a minimum close to 0
        assert!(de_result.fun < 1e-4);
        assert!(sa_result.fun < 0.5); // SA is less accurate
        assert!(da_result.fun < 1e-4);
    }
}
