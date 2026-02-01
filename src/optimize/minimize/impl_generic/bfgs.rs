//! BFGS quasi-Newton method for multivariate minimization.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, backtracking_line_search_tensor};
use super::utils::{SINGULAR_THRESHOLD, finite_difference_gradient, tensor_norm};

/// BFGS quasi-Newton method for minimization using tensors.
///
/// All operations use tensor ops to stay on device (CPU/GPU).
pub fn bfgs_impl<R, C, F>(
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
            context: "bfgs: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("bfgs: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut grad = finite_difference_gradient(client, &f, &x, fx, options.eps).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("bfgs: gradient - {}", e),
        }
    })?;
    nfev += n;

    // Initialize inverse Hessian approximation as identity matrix tensor [n, n]
    let mut h_inv = create_identity_matrix::<R, C>(client, n)?;

    for iter in 0..options.max_iter {
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs: grad norm - {}", e),
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

        // Compute search direction: p = -H_inv @ grad
        // Reshape grad to [n, 1] for matmul, then reshape result back to [n]
        let grad_col = grad
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: grad reshape - {}", e),
            })?;
        let h_grad =
            client
                .matmul(&h_inv, &grad_col)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: h_inv @ grad - {}", e),
                })?;
        let h_grad_flat = h_grad
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: h_grad reshape - {}", e),
            })?;
        let p =
            client
                .mul_scalar(&h_grad_flat, -1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: negate direction - {}", e),
                })?;

        // Line search
        let (x_new, fx_new, evals) =
            backtracking_line_search_tensor(client, &f, &x, &p, fx, &grad)?;
        nfev += evals;

        // Compute step difference: s = x_new - x
        let s = client
            .sub(&x_new, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: s = x_new - x - {}", e),
            })?;

        // Check convergence based on step size
        let s_norm = tensor_norm(client, &s).map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs: s norm - {}", e),
        })?;

        if s_norm < options.x_tol || (fx - fx_new).abs() < options.f_tol {
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
                message: format!("bfgs: new gradient - {}", e),
            })?;
        nfev += n;

        // Compute gradient difference: y = grad_new - grad
        let y = client
            .sub(&grad_new, &grad)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: y = grad_new - grad - {}", e),
            })?;

        // BFGS update: H_inv = (I - rho * s @ y.T) @ H_inv @ (I - rho * y @ s.T) + rho * s @ s.T
        // where rho = 1 / (y.T @ s)

        // Compute y.T @ s (inner product)
        let ys = tensor_inner_product(client, &y, &s)?;

        if ys.abs() > SINGULAR_THRESHOLD {
            let rho = 1.0 / ys;

            // Reshape s and y to column vectors for outer products
            let s_col = s
                .reshape(&[n, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: s reshape - {}", e),
                })?;
            let y_col = y
                .reshape(&[n, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: y reshape - {}", e),
                })?;
            let s_row = s
                .reshape(&[1, n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: s row reshape - {}", e),
                })?;
            let y_row = y
                .reshape(&[1, n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: y row reshape - {}", e),
                })?;

            // Compute s @ s.T (outer product)
            let s_st =
                client
                    .matmul(&s_col, &s_row)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: s @ s.T - {}", e),
                    })?;

            // Compute H_inv @ y (as column vector)
            let h_y = client
                .matmul(&h_inv, &y_col)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: H_inv @ y - {}", e),
                })?;

            // Compute y.T @ H_inv @ y
            let y_row_h = y_row.clone();
            let yhy =
                {
                    let yt_h = client.matmul(&y_row_h, &h_inv).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("bfgs: y.T @ H_inv - {}", e),
                        }
                    })?;
                    let yt_h_y = client.matmul(&yt_h, &y_col).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("bfgs: y.T @ H_inv @ y - {}", e),
                        }
                    })?;
                    // yt_h_y is [1,1], extract scalar
                    let vals: Vec<f64> = yt_h_y.to_vec();
                    vals[0]
                };

            // BFGS update formula (Sherman-Morrison variant):
            // H_inv_new = H_inv + rho * (1 + rho * y.T @ H_inv @ y) * s @ s.T
            //           - rho * (s @ (H_inv @ y).T + H_inv @ y @ s.T)
            //
            // Which simplifies to:
            // H_inv_new = H_inv + rho * (1 + rho * yHy) * s @ s.T - rho * (s @ h_y.T + h_y @ s.T)

            let h_y_row = h_y
                .reshape(&[1, n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: h_y row reshape - {}", e),
                })?;

            // s @ h_y.T
            let s_hyt =
                client
                    .matmul(&s_col, &h_y_row)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: s @ h_y.T - {}", e),
                    })?;

            // h_y @ s.T
            let hy_st = client
                .matmul(&h_y, &s_row)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs: h_y @ s.T - {}", e),
                })?;

            // term1 = rho * (1 + rho * yHy) * s @ s.T
            let coeff1 = rho * (1.0 + rho * yhy);
            let term1 =
                client
                    .mul_scalar(&s_st, coeff1)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: term1 - {}", e),
                    })?;

            // term2 = rho * (s @ h_y.T + h_y @ s.T)
            let sum_outer =
                client
                    .add(&s_hyt, &hy_st)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: s_hyt + hy_st - {}", e),
                    })?;
            let term2 =
                client
                    .mul_scalar(&sum_outer, rho)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: term2 - {}", e),
                    })?;

            // H_inv_new = H_inv + term1 - term2
            let h_plus_term1 =
                client
                    .add(&h_inv, &term1)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: H + term1 - {}", e),
                    })?;
            h_inv =
                client
                    .sub(&h_plus_term1, &term2)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("bfgs: H + term1 - term2 - {}", e),
                    })?;
        }

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

/// Create an n x n identity matrix using tensor ops.
fn create_identity_matrix<R, C>(client: &C, n: usize) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs: create identity - {}", e),
        })
}

/// Compute inner product of two vectors: sum(a * b)
fn tensor_inner_product<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n = a.shape()[0];

    // Reshape a to [1, n] and b to [n, 1] for matmul
    let a_row = a
        .reshape(&[1, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("inner_product: a reshape - {}", e),
        })?;
    let b_col = b
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("inner_product: b reshape - {}", e),
        })?;

    // Compute [1,n] @ [n,1] = [1,1]
    let result = client
        .matmul(&a_row, &b_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("inner_product: matmul - {}", e),
        })?;

    // Extract scalar - this is the only device transfer (for convergence check)
    let vals: Vec<f64> = result.to_vec();
    Ok(vals[0])
}
