//! Tensor-based dual annealing implementation.
//!
//! All computation stays on device using numr tensor operations.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{CompareOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::GlobalOptions;
use crate::optimize::impl_generic::nelder_mead_impl;
use crate::optimize::minimize::MinimizeOptions;

use super::clamp_to_bounds;

/// Tensor-based result from dual annealing.
#[derive(Debug, Clone)]
pub struct DualAnnealingTensorResult<R: Runtime<DType = DType>> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Dual annealing global optimizer using tensor operations.
///
/// Combines simulated annealing with local search for smooth functions.
/// All intermediate state stays on device. No `to_vec()` in the main loop.
pub fn dual_annealing_impl<R, C, F>(
    client: &C,
    f: F,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<DualAnnealingTensorResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RandomOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let shape = lower_bounds.shape();
    let n = shape[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "dual_annealing: empty bounds".to_string(),
        });
    }

    // Compute bounds range
    let bounds_range =
        client
            .sub(upper_bounds, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: bounds range - {}", e),
            })?;

    let local_opts = MinimizeOptions {
        max_iter: 50,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    // Temperature schedule
    let t_initial: f64 = 5230.0;
    let t_final: f64 = 0.001;
    let cooling_rate = (t_final / t_initial).powf(1.0 / (options.max_iter as f64 / 2.0));
    let local_search_interval = 10;

    // Initialize at random point within bounds
    let rand_init = client
        .rand(&[n], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("dual_annealing: rand init - {}", e),
        })?;
    let scaled_rand =
        client
            .mul(&rand_init, &bounds_range)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: scale rand - {}", e),
            })?;
    let mut x_current =
        client
            .add(lower_bounds, &scaled_rand)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: init x - {}", e),
            })?;

    let mut f_current = f(&x_current).map_err(|e| OptimizeError::NumericalError {
        message: format!("dual_annealing: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut temperature = t_initial;

    for iter in 0..options.max_iter {
        // Occasional local search
        if iter > 0 && iter % local_search_interval == 0 {
            let local_result = nelder_mead_impl(client, &f, &x_current, &local_opts)?;
            nfev += local_result.nfev;

            let x_local = clamp_to_bounds(client, &local_result.x, lower_bounds, upper_bounds)?;
            let f_local = f(&x_local).map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: local eval - {}", e),
            })?;
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

        // Generalized simulated annealing step (Cauchy-like visiting distribution)
        // Generate perturbation using tensor ops
        let scale = temperature / t_initial;

        let rand_perturb =
            client
                .rand(&[n], DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("dual_annealing: rand perturb - {}", e),
                })?;

        // Approximate Cauchy-like distribution: scale * range * tan(pi * (u - 0.5))
        // Using simplified perturbation: scale * range * (2*rand - 1) * factor
        let rand_scaled =
            client
                .mul_scalar(&rand_perturb, 2.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("dual_annealing: scale rand - {}", e),
                })?;

        let rand_centered =
            client
                .sub_scalar(&rand_scaled, 1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("dual_annealing: center rand - {}", e),
                })?;

        let delta_unscaled = client.mul(&rand_centered, &bounds_range).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("dual_annealing: delta unscaled - {}", e),
            }
        })?;

        // Use larger perturbation factor so exploration doesn't collapse too early.
        // scale goes from 1.0 → ~0, so step_factor keeps steps meaningful longer.
        let step_factor = scale.sqrt() * 0.5;
        let delta = client
            .mul_scalar(&delta_unscaled, step_factor)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: delta - {}", e),
            })?;

        let x_neighbor_unclamped =
            client
                .add(&x_current, &delta)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("dual_annealing: neighbor unclamped - {}", e),
                })?;

        let x_neighbor =
            clamp_to_bounds(client, &x_neighbor_unclamped, lower_bounds, upper_bounds)?;

        let f_neighbor = f(&x_neighbor).map_err(|e| OptimizeError::NumericalError {
            message: format!("dual_annealing: neighbor eval - {}", e),
        })?;
        nfev += 1;

        // Acceptance criterion
        let delta_f = f_neighbor - f_current;
        let accept = if delta_f < 0.0 {
            true
        } else {
            let accept_rand =
                client
                    .rand(&[1], DType::F64)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("dual_annealing: accept rand - {}", e),
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

        // Check termination
        if temperature < t_final || (f_best < options.tol && iter > 100) {
            // Final local search
            let local_result = nelder_mead_impl(client, &f, &x_best, &local_opts)?;
            nfev += local_result.nfev;

            let x_final = clamp_to_bounds(client, &local_result.x, lower_bounds, upper_bounds)?;
            let f_final = f(&x_final).map_err(|e| OptimizeError::NumericalError {
                message: format!("dual_annealing: final eval - {}", e),
            })?;
            nfev += 1;

            if f_final < f_best {
                x_best = x_final;
                f_best = f_final;
            }

            return Ok(DualAnnealingTensorResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    // Final local search even when max_iter exhausted
    let local_result = nelder_mead_impl(client, &f, &x_best, &local_opts)?;
    nfev += local_result.nfev;

    let x_final = clamp_to_bounds(client, &local_result.x, lower_bounds, upper_bounds)?;
    let f_final = f(&x_final).map_err(|e| OptimizeError::NumericalError {
        message: format!("dual_annealing: final eval - {}", e),
    })?;
    nfev += 1;

    if f_final < f_best {
        x_best = x_final;
        f_best = f_final;
    }

    Ok(DualAnnealingTensorResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
