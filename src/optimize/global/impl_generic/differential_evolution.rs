//! Tensor-based differential evolution implementation.
//!
//! Population stored as Vec<Tensor<R>> where each element is shape [n].
//! All tensor operations stay on device.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::GlobalOptions;

use super::{clamp_to_bounds, validate_bounds};

/// Tensor-based result from differential evolution.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionTensorResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Differential Evolution global optimizer using tensor operations.
///
/// Population is stored as Vec<Tensor<R>> where each is shape [n].
/// All operations stay on device - no to_vec()/from_slice() in loops.
pub fn differential_evolution_impl<R, C, F>(
    client: &C,
    f: F,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<DifferentialEvolutionTensorResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = lower_bounds.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "differential_evolution: empty bounds".to_string(),
        });
    }

    // Validate bounds using tensor ops
    validate_bounds(client, lower_bounds, upper_bounds)?;

    // DE parameters
    let pop_size = (15 * n).max(25);
    let f_scale = 0.8;
    let cr = 0.9;

    // Compute bounds range (stays on device)
    let bounds_range =
        client
            .sub(upper_bounds, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: bounds range - {}", e),
            })?;

    // Initialize population as Vec<Tensor<R>> - each element is shape [n]
    let mut population = init_population(client, lower_bounds, &bounds_range, pop_size, n)?;

    // Evaluate initial population - must iterate (function returns scalar)
    let mut fitness = evaluate_population(&f, &population)?;
    let mut nfev = pop_size;

    // Find best individual
    let (mut best_idx, mut best_fitness) = find_best(&fitness);

    // Pre-create index tensor for crossover (reused each iteration)
    let indices = client.arange(0.0, n as f64, 1.0, DType::F64).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("de: create indices - {}", e),
        }
    })?;

    for iter in 0..options.max_iter {
        // Check convergence
        let fitness_range = compute_fitness_range(&fitness);
        if fitness_range < options.tol {
            return Ok(DifferentialEvolutionTensorResult {
                x: population[best_idx].clone(),
                fun: best_fitness,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // DE iteration: for each individual, create trial and possibly replace
        for i in 0..pop_size {
            // Select three distinct random individuals (not i)
            let (r0, r1, r2) = select_random_indices::<R, C>(client, pop_size, i)?;

            // Get references to individuals for mutation
            let x_r0 = &population[r0];
            let x_r1 = &population[r1];
            let x_r2 = &population[r2];
            let x_i = &population[i];

            // Mutant: x_r0 + f_scale * (x_r1 - x_r2)
            let diff = client
                .sub(x_r1, x_r2)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("de: diff - {}", e),
                })?;
            let scaled_diff =
                client
                    .mul_scalar(&diff, f_scale)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("de: scaled diff - {}", e),
                    })?;
            let mutant_unclamped =
                client
                    .add(x_r0, &scaled_diff)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("de: mutant - {}", e),
                    })?;

            // Clamp to bounds
            let mutant = clamp_to_bounds(client, &mutant_unclamped, lower_bounds, upper_bounds)?;

            // Crossover: create trial vector using tensor ops
            let trial = crossover(client, x_i, &mutant, cr, n, &indices)?;

            // Evaluate trial
            let trial_fitness = f(&trial).map_err(|e| OptimizeError::NumericalError {
                message: format!("de: evaluation - {}", e),
            })?;
            nfev += 1;

            // Selection: if trial is better, replace
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

    Ok(DifferentialEvolutionTensorResult {
        x: population[best_idx].clone(),
        fun: best_fitness,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Initialize population as Vec<Tensor<R>> with uniform random within bounds.
fn init_population<R, C>(
    client: &C,
    lower: &Tensor<R>,
    range: &Tensor<R>,
    pop_size: usize,
    n: usize,
) -> OptimizeResult<Vec<Tensor<R>>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut population = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        // Generate random [n] in [0, 1)
        let rand_ind =
            client
                .rand(&[n], DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("de: rand individual - {}", e),
                })?;

        // individual = lower + rand * range
        let scaled = client
            .mul(&rand_ind, range)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: scale individual - {}", e),
            })?;
        let individual = client
            .add(lower, &scaled)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: init individual - {}", e),
            })?;

        population.push(individual);
    }

    Ok(population)
}

/// Evaluate all individuals in population.
fn evaluate_population<R, F>(f: &F, population: &[Tensor<R>]) -> OptimizeResult<Vec<f64>>
where
    R: Runtime,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut fitness = Vec::with_capacity(population.len());
    for individual in population {
        let fit = f(individual).map_err(|e| OptimizeError::NumericalError {
            message: format!("de: initial evaluation - {}", e),
        })?;
        fitness.push(fit);
    }
    Ok(fitness)
}

/// Binomial crossover between target and mutant using tensor operations.
///
/// Uses tensor ops to select elements: where (rand < cr OR index == j_rand), use mutant, else target.
/// No to_vec()/from_slice() - all computation stays on device.
fn crossover<R, C>(
    client: &C,
    target: &Tensor<R>,
    mutant: &Tensor<R>,
    cr: f64,
    n: usize,
    indices: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    // Generate random mask [n] in [0, 1)
    let rand_mask = client
        .rand(&[n], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: crossover rand - {}", e),
        })?;

    // Create threshold tensor filled with cr
    let cr_tensor =
        client
            .fill(&[n], cr, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: cr tensor - {}", e),
            })?;

    // mask1: where rand_mask < cr (use mutant) - returns F64 (0.0 or 1.0)
    let lt_mask = client
        .lt(&rand_mask, &cr_tensor)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: lt comparison - {}", e),
        })?;

    // Generate j_rand: random index in [0, n) - single scalar extraction is acceptable
    let rand_idx = client
        .rand(&[1], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: j_rand - {}", e),
        })?;
    let j_rand_val: Vec<f64> = rand_idx.to_vec();
    let j_rand = (j_rand_val[0] * n as f64) as usize;

    // Create j_rand tensor for comparison
    let j_rand_tensor = client.fill(&[n], j_rand as f64, DType::F64).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("de: j_rand tensor - {}", e),
        }
    })?;

    // mask2: where index == j_rand (one-hot at j_rand position) - returns F64 (0.0 or 1.0)
    let j_rand_mask =
        client
            .eq(indices, &j_rand_tensor)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: j_rand mask - {}", e),
            })?;

    // Combined mask: use maximum for OR-like behavior with F64 (0.0/1.0) masks
    let combined_f64 =
        client
            .maximum(&lt_mask, &j_rand_mask)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: combine masks - {}", e),
            })?;

    // Cast to U8 for where_cond (which expects U8 boolean mask)
    let combined_mask =
        client
            .cast(&combined_f64, DType::U8)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("de: cast mask - {}", e),
            })?;

    // Select: where mask is true use mutant, else use target
    client
        .where_cond(&combined_mask, mutant, target)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: crossover select - {}", e),
        })
}

/// Select 3 distinct random indices, none equal to exclude.
///
/// Uses tensor random ops to generate random values, then extracts as scalars.
/// Single-scalar extraction is acceptable (not arrays).
fn select_random_indices<R, C>(
    client: &C,
    pop_size: usize,
    exclude: usize,
) -> OptimizeResult<(usize, usize, usize)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // Generate 6 random values to select 3 distinct indices (with buffer for collision retry).
    // We need 3 unique indices != exclude; 6 gives ~99% success rate for typical pop_size >= 10.
    let rand_vals = client
        .rand(&[6], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: select random - {}", e),
        })?;
    let vals: Vec<f64> = rand_vals.to_vec();

    let mut selected = Vec::with_capacity(3);
    let mut idx = 0;

    while selected.len() < 3 && idx < 6 {
        let candidate = (vals[idx] * pop_size as f64) as usize % pop_size;
        if candidate != exclude && !selected.contains(&candidate) {
            selected.push(candidate);
        }
        idx += 1;
    }

    // Fallback if we didn't get enough random unique indices
    if selected.len() < 3 {
        for k in 0..pop_size {
            if k != exclude && !selected.contains(&k) {
                selected.push(k);
                if selected.len() >= 3 {
                    break;
                }
            }
        }
    }

    Ok((selected[0], selected[1], selected[2]))
}

fn find_best(fitness: &[f64]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_val = fitness[0];
    for (i, &f) in fitness.iter().enumerate() {
        if f < best_val {
            best_val = f;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

fn compute_fitness_range(fitness: &[f64]) -> f64 {
    let max = fitness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}
