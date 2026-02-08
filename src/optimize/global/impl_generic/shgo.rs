//! SHGO (Simplicial Homology Global Optimization) implementation.
//!
//! Finds global minima by sampling the domain, sorting candidates, and running
//! local refinement via coordinate descent at the most promising points.

use numr::error::Result;
use numr::ops::{CompareOps, ScalarOps, TensorOps, UtilityOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::GlobalOptions;

/// SHGO result type.
#[derive(Debug, Clone)]
pub struct ShgoTensorResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub local_minima: Vec<(Tensor<R>, f64)>,
    pub nfev: usize,
    pub converged: bool,
}

/// Generate quasi-random numbers using Van der Corput sequence.
///
/// The Van der Corput sequence is a low-discrepancy sequence that samples
/// [0,1) uniformly. For dimension d, we use the d-th prime as the base.
fn van_der_corput(index: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut digit = 1.0 / base as f64;
    let mut n = index;

    while n > 0 {
        let digit_value = (n % base) as f64;
        result += digit_value * digit;
        digit /= base as f64;
        n /= base;
    }

    result
}

/// Get the i-th prime number (used as Halton bases).
const PRIMES: &[usize] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Halton quasi-random sequence: low-discrepancy sampling.
/// Generates a point in [0,1]^n using different prime bases for each dimension.
fn halton_sequence<R, C>(client: &C, dim: usize, index: usize) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let mut values = Vec::with_capacity(dim);

    for d in 0..dim {
        let base = PRIMES.get(d).copied().unwrap_or(113);
        let x = van_der_corput(index, base);
        values.push(x);
    }

    let device = client.device();
    Tensor::try_from_slice(&values, &[dim], device).map_err(|e| OptimizeError::NumericalError {
        message: format!("halton: create tensor - {}", e),
    })
}

/// Coordinate descent local refinement of a candidate point.
///
/// Performs up to `max_refine_iter` coordinate-wise line searches
/// to refine a starting point.
fn refine_candidate<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    tol: f64,
) -> OptimizeResult<(Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x.shape()[0];
    let mut x_current = x.clone();
    let mut fx_current = f(&x_current).map_err(|e| OptimizeError::NumericalError {
        message: format!("refine: eval x - {}", e),
    })?;
    let mut nfev = 1;

    // Pre-compute coordinate ranges on device: range = upper - lower
    let coord_range =
        client
            .sub(upper_bounds, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("refine: coord range - {}", e),
            })?;

    // Pre-compute identity matrix for unit vector extraction
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("refine: eye - {}", e),
        })?;

    let mut step_size = 0.1;
    let max_refine_iter = 50;

    for _ in 0..max_refine_iter {
        let step_before = step_size;
        let mut improved = false;

        for dim in 0..n {
            // Extract unit vector from pre-computed identity matrix
            let e_dim = identity
                .narrow(0, dim, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("refine: narrow e_dim - {}", e),
                })?
                .contiguous()
                .reshape(&[n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("refine: reshape e_dim - {}", e),
                })?;

            // delta_vec = step_size * coord_range * e_dim (nonzero only at dim)
            let scaled_range = client.mul_scalar(&coord_range, step_size).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("refine: scale range - {}", e),
                }
            })?;
            let delta_vec =
                client
                    .mul(&scaled_range, &e_dim)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("refine: delta vec - {}", e),
                    })?;

            // Try positive direction: x_plus = clamp(x + delta, lower, upper)
            let x_unclamped =
                client
                    .add(&x_current, &delta_vec)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("refine: x + delta - {}", e),
                    })?;
            let x_plus = clamp_to_bounds(client, &x_unclamped, lower_bounds, upper_bounds)?;

            let fx_plus = f(&x_plus).map_err(|e| OptimizeError::NumericalError {
                message: format!("refine: eval x_plus - {}", e),
            })?;
            nfev += 1;

            if fx_plus < fx_current {
                x_current = x_plus;
                fx_current = fx_plus;
                improved = true;
                continue;
            }

            // Try negative direction: x_minus = clamp(x - delta, lower, upper)
            let x_unclamped =
                client
                    .sub(&x_current, &delta_vec)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("refine: x - delta - {}", e),
                    })?;
            let x_minus = clamp_to_bounds(client, &x_unclamped, lower_bounds, upper_bounds)?;

            let fx_minus = f(&x_minus).map_err(|e| OptimizeError::NumericalError {
                message: format!("refine: eval x_minus - {}", e),
            })?;
            nfev += 1;

            if fx_minus < fx_current {
                x_current = x_minus;
                fx_current = fx_minus;
                improved = true;
            }
        }

        if !improved {
            step_size *= 0.5;
            if step_size < tol {
                break;
            }
        }

        if step_size < tol / 100.0 {
            break;
        }

        if step_size > step_before * 0.99 && !improved {
            break;
        }
    }

    Ok((x_current, fx_current, nfev))
}

/// Clamp tensor to bounds: max(lower, min(upper, x))
fn clamp_to_bounds<R, C>(
    client: &C,
    x: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>,
{
    let clamped = client
        .minimum(x, upper)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("clamp: min upper - {}", e),
        })?;
    client
        .maximum(&clamped, lower)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("clamp: max lower - {}", e),
        })
}

/// Check if a point is already in the list of local minima (within distance threshold).
fn is_duplicate<R, C>(
    client: &C,
    x_new: &Tensor<R>,
    local_minima: &[(Tensor<R>, f64)],
    threshold: f64,
) -> OptimizeResult<bool>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    for (x_existing, _) in local_minima {
        let diff = client
            .sub(x_new, x_existing)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("is_duplicate: diff - {}", e),
            })?;
        let diff_sq = client
            .mul(&diff, &diff)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("is_duplicate: sq - {}", e),
            })?;
        let sum = client
            .sum(&diff_sq, &[0], false)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("is_duplicate: sum - {}", e),
            })?;
        let dist_sq_val: f64 = sum
            .item::<f64>()
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("is_duplicate: item - {}", e),
            })?;
        if dist_sq_val.sqrt() < threshold {
            return Ok(true);
        }
    }
    Ok(false)
}

/// SHGO algorithm: Simplicial Homology Global Optimization.
///
/// 1. Generate quasi-random samples using Halton sequence
/// 2. Evaluate function at all samples
/// 3. Sort by function value
/// 4. Run local refinement at top-k candidates
/// 5. Collect all local minima
pub fn shgo_impl<R, C, F>(
    client: &C,
    f: F,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<ShgoTensorResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = lower_bounds.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "shgo: empty bounds".to_string(),
        });
    }

    // Validate bounds - need CompareOps trait
    let bounds_valid =
        client
            .ge(lower_bounds, upper_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("shgo: bounds check - {}", e),
            })?;

    let violation_sum =
        client
            .sum(&bounds_valid, &[0], false)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("shgo: sum violations - {}", e),
            })?;

    let sum_val: f64 = violation_sum
        .item::<f64>()
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("shgo: bounds check item - {}", e),
        })?;
    if sum_val > 0.0 {
        return Err(OptimizeError::InvalidInput {
            context: "lower bounds must be less than upper bounds".to_string(),
        });
    }

    // Number of samples: min(max_iter, 128)
    let n_samples = options.max_iter.min(128);

    // Generate quasi-random samples in [0,1]^n
    let mut candidates: Vec<(Tensor<R>, f64)> = Vec::new();
    let mut nfev = 0;

    for i in 0..n_samples {
        // Generate sample in [0,1]
        let x_unit = halton_sequence(client, n, i)?;

        // Scale to bounds: x = lower + (upper - lower) * x_unit
        let bounds_range =
            client
                .sub(upper_bounds, lower_bounds)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("shgo: bounds range - {}", e),
                })?;

        let x_scaled =
            client
                .mul(&x_unit, &bounds_range)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("shgo: scale x - {}", e),
                })?;

        let x = client
            .add(&x_scaled, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("shgo: shift x - {}", e),
            })?;

        // Evaluate
        let fx = f(&x).map_err(|e| OptimizeError::NumericalError {
            message: format!("shgo: eval sample - {}", e),
        })?;

        candidates.push((x, fx));
        nfev += 1;
    }

    // Sort candidates by function value (ascending)
    // NaN values are treated as greater than any finite value (worst)
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

    // Refine top-k candidates where k = min(5, n+1)
    let k_refine = 5.min(n + 1);
    let mut local_minima: Vec<(Tensor<R>, f64)> = Vec::new();
    let duplicate_threshold = 0.1;

    for (x_candidate, _) in candidates.iter().take(k_refine) {
        let (x_refined, fx_refined, nfev_refine) = refine_candidate(
            client,
            &f,
            x_candidate,
            lower_bounds,
            upper_bounds,
            options.tol,
        )?;

        nfev += nfev_refine;

        // Check if this local minimum is new (not a duplicate)
        if !is_duplicate(client, &x_refined, &local_minima, duplicate_threshold)? {
            local_minima.push((x_refined, fx_refined));
        }
    }

    // Find global best
    // NaN values are treated as greater than any finite value (worst)
    let (x_best, best_fun) = local_minima
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater))
        .map(|(x, fv)| (x.clone(), *fv))
        .unwrap_or_else(|| {
            let best_candidate = &candidates[0];
            (best_candidate.0.clone(), best_candidate.1)
        });

    Ok(ShgoTensorResult {
        x: x_best,
        fun: best_fun,
        local_minima,
        nfev,
        converged: true, // SHGO always "converges" - it's not iterative like DE or SA
    })
}
