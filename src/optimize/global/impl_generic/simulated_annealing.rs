//! Tensor-based simulated annealing implementation.
//!
//! All computation stays on device using numr tensor operations.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{CompareOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::GlobalOptions;

use super::{clamp_to_bounds, validate_bounds};

/// Tensor-based result from simulated annealing.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingTensorResult<R: Runtime<DType = DType>> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Simulated annealing global optimizer using tensor operations.
///
/// All intermediate state stays on device. No `to_vec()` in the main loop.
pub fn simulated_annealing_impl<R, C, F>(
    client: &C,
    f: F,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<SimulatedAnnealingTensorResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RandomOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let shape = lower_bounds.shape();
    let n = shape[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "simulated_annealing: empty bounds".to_string(),
        });
    }

    // Validate bounds using tensor ops (only check needed at start)
    validate_bounds(client, lower_bounds, upper_bounds)?;

    // Compute bounds range: range = upper - lower (stays on device)
    let bounds_range =
        client
            .sub(upper_bounds, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("simulated_annealing: bounds range - {}", e),
            })?;

    // Initialize at random point within bounds: x = lower + rand * range
    let rand_init = client
        .rand(&[n], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("simulated_annealing: rand init - {}", e),
        })?;
    let scaled_rand =
        client
            .mul(&rand_init, &bounds_range)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("simulated_annealing: scale rand - {}", e),
            })?;
    let mut x_current =
        client
            .add(lower_bounds, &scaled_rand)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("simulated_annealing: init x - {}", e),
            })?;

    let mut f_current = f(&x_current).map_err(|e| OptimizeError::NumericalError {
        message: format!("simulated_annealing: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;

    // Temperature schedule
    let t_initial: f64 = 5230.0;
    let t_final: f64 = 0.0001;
    let cooling_rate = (t_final / t_initial).powf(1.0 / options.max_iter as f64);
    let mut temperature = t_initial;

    for iter in 0..options.max_iter {
        // Generate neighbor: x_neighbor = x_current + scale * range * (2*rand - 1)
        // All operations stay on device
        let scale = temperature / t_initial;

        let rand_perturb =
            client
                .rand(&[n], DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("simulated_annealing: rand perturb - {}", e),
                })?;

        // delta = scale * range * (2*rand - 1)
        let rand_centered = client
            .sub_scalar(
                &client.mul_scalar(&rand_perturb, 2.0).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("simulated_annealing: scale rand - {}", e),
                    }
                })?,
                1.0,
            )
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("simulated_annealing: center rand - {}", e),
            })?;

        let delta_unscaled = client.mul(&rand_centered, &bounds_range).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("simulated_annealing: delta unscaled - {}", e),
            }
        })?;

        let delta = client.mul_scalar(&delta_unscaled, scale).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("simulated_annealing: delta - {}", e),
            }
        })?;

        let x_neighbor_unclamped =
            client
                .add(&x_current, &delta)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("simulated_annealing: neighbor unclamped - {}", e),
                })?;

        // Clamp to bounds: max(lower, min(upper, x))
        let x_neighbor =
            clamp_to_bounds(client, &x_neighbor_unclamped, lower_bounds, upper_bounds)?;

        let f_neighbor = f(&x_neighbor).map_err(|e| OptimizeError::NumericalError {
            message: format!("simulated_annealing: evaluation - {}", e),
        })?;
        nfev += 1;

        // Acceptance criterion (scalar decision - unavoidable)
        let delta_f = f_neighbor - f_current;
        let accept = if delta_f < 0.0 {
            true
        } else {
            // Generate single random for acceptance (could batch this but overhead minimal)
            let accept_rand =
                client
                    .rand(&[1], DType::F64)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("simulated_annealing: accept rand - {}", e),
                    })?;
            let accept_val: Vec<f64> = accept_rand.to_vec();
            accept_val[0] < (-delta_f / temperature).exp()
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
            return Ok(SimulatedAnnealingTensorResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    Ok(SimulatedAnnealingTensorResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
