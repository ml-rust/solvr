//! Conjugate gradient method for multivariate minimization.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, backtracking_line_search_tensor};
use super::utils::{SINGULAR_THRESHOLD, finite_difference_gradient, tensor_dot, tensor_norm};

/// Conjugate gradient method for minimization using tensors.
///
/// All operations use tensor ops to stay on device (CPU/GPU).
pub fn conjugate_gradient_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &MinimizeOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "conjugate_gradient: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("cg: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut grad = finite_difference_gradient(client, &f, &x, fx, options.eps).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("cg: gradient - {}", e),
        }
    })?;
    nfev += n;

    // Initial search direction is negative gradient: p = -grad
    let mut p = client
        .mul_scalar(&grad, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("cg: initial direction - {}", e),
        })?;

    for iter in 0..options.max_iter {
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("cg: grad norm - {}", e),
        })?;

        if grad_norm < options.g_tol {
            return Ok(TensorMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Line search
        let (x_new, fx_new, evals) =
            backtracking_line_search_tensor(client, &f, &x, &p, fx, &grad)?;
        nfev += evals;

        // Check convergence
        let dx = client
            .sub(&x_new, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cg: dx - {}", e),
            })?;
        let dx_norm = tensor_norm(client, &dx).map_err(|e| OptimizeError::NumericalError {
            message: format!("cg: dx norm - {}", e),
        })?;

        if dx_norm < options.x_tol || (fx - fx_new).abs() < options.f_tol {
            return Ok(TensorMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient(client, &f, &x_new, fx_new, options.eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cg: new gradient - {}", e),
            })?;
        nfev += n;

        // Polak-Ribieri beta = grad_new.T @ (grad_new - grad) / (grad.T @ grad)
        let grad_old_norm_sq =
            tensor_dot(client, &grad, &grad).map_err(|e| OptimizeError::NumericalError {
                message: format!("cg: grad norm sq - {}", e),
            })?;

        let grad_diff =
            client
                .sub(&grad_new, &grad)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("cg: grad diff - {}", e),
                })?;
        let beta_num = tensor_dot(client, &grad_new, &grad_diff).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("cg: beta num - {}", e),
            }
        })?;

        let beta = if grad_old_norm_sq > SINGULAR_THRESHOLD {
            (beta_num / grad_old_norm_sq).max(0.0) // Restart if negative
        } else {
            0.0
        };

        // New search direction: p = -grad_new + beta * p
        let neg_grad_new =
            client
                .mul_scalar(&grad_new, -1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("cg: neg grad new - {}", e),
                })?;
        let beta_p = client
            .mul_scalar(&p, beta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cg: beta * p - {}", e),
            })?;
        p = client
            .add(&neg_grad_new, &beta_p)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cg: new direction - {}", e),
            })?;

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    Ok(TensorMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
