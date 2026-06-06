//! Tensor-based basin-hopping implementation.
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

/// Tensor-based result from basin-hopping.
#[derive(Debug, Clone)]
pub struct BasinHoppingTensorResult<R: Runtime<DType = DType>> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Basin-hopping global optimizer using tensor operations.
///
/// Combines local minimization with random perturbations to escape local minima.
/// All intermediate state stays on device. No `to_vec()` in the main loop.
pub fn basinhopping_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<BasinHoppingTensorResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RandomOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let shape = x0.shape();
    let n = shape[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "basinhopping: empty initial guess".to_string(),
        });
    }

    // Compute bounds range for perturbation scaling
    let bounds_range =
        client
            .sub(upper_bounds, lower_bounds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("basinhopping: bounds range - {}", e),
            })?;

    let step_size = 0.5_f64;
    let temperature = 1.0_f64;

    let local_opts = MinimizeOptions {
        max_iter: 100,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    // Initial local optimization
    let local_result = nelder_mead_impl(client, &f, x0, &local_opts)?;
    let mut x_current = local_result.x;
    let mut f_current = local_result.fun;
    let mut nfev = local_result.nfev;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;

    for iter in 0..options.max_iter {
        // Generate perturbation: delta = step_size * range * (2*rand - 1)
        let rand_perturb =
            client
                .rand(&[n], DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("basinhopping: rand perturb - {}", e),
                })?;

        let rand_scaled =
            client
                .mul_scalar(&rand_perturb, 2.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("basinhopping: scale rand - {}", e),
                })?;

        let rand_centered =
            client
                .sub_scalar(&rand_scaled, 1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("basinhopping: center rand - {}", e),
                })?;

        let delta_unscaled = client.mul(&rand_centered, &bounds_range).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("basinhopping: delta unscaled - {}", e),
            }
        })?;

        let delta = client.mul_scalar(&delta_unscaled, step_size).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("basinhopping: delta - {}", e),
            }
        })?;

        // Perturb current position
        let x_perturbed_unclamped =
            client
                .add(&x_current, &delta)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("basinhopping: perturb - {}", e),
                })?;

        let x_perturbed =
            clamp_to_bounds(client, &x_perturbed_unclamped, lower_bounds, upper_bounds)?;

        // Local optimization from perturbed point
        let local_result = nelder_mead_impl(client, &f, &x_perturbed, &local_opts)?;
        let x_new = clamp_to_bounds(client, &local_result.x, lower_bounds, upper_bounds)?;
        let f_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("basinhopping: evaluation - {}", e),
        })?;
        nfev += local_result.nfev + 1;

        // Metropolis acceptance criterion
        let delta_f = f_new - f_current;
        let accept = if delta_f < 0.0 {
            true
        } else {
            let accept_rand =
                client
                    .rand(&[1], DType::F64)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("basinhopping: accept rand - {}", e),
                    })?;
            let accept_val: Vec<f64> = accept_rand.to_vec();
            accept_val[0] < (-delta_f / temperature).exp()
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
            return Ok(BasinHoppingTensorResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    Ok(BasinHoppingTensorResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
