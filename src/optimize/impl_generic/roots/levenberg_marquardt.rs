//! Levenberg-Marquardt algorithm for systems of nonlinear equations using tensors.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::utils::SINGULAR_THRESHOLD;
use crate::optimize::roots::RootOptions;

use super::TensorRootResult;
use super::newton::finite_difference_jacobian_tensor;
use crate::optimize::impl_generic::utils::tensor_norm;

/// Levenberg-Marquardt algorithm for systems of nonlinear equations.
///
/// A damped Newton method that interpolates between Newton's method and
/// gradient descent. More robust when initial guess is far from solution.
pub fn levenberg_marquardt_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &RootOptions,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "levenberg_marquardt: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("levenberg_marquardt: initial evaluation - {}", e),
    })?;

    if fx.shape()[0] != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "levenberg_marquardt: function returns {} values but input has {} dimensions",
                fx.shape()[0],
                n
            ),
        });
    }

    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let lambda_min = SINGULAR_THRESHOLD;
    let lambda_max = 1e10;

    for iter in 0..options.max_iter {
        let res_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
            message: format!("levenberg_marquardt: norm - {}", e),
        })?;

        if res_norm < options.tol {
            return Ok(TensorRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Compute Jacobian
        let jacobian = finite_difference_jacobian_tensor(client, &f, &x, &fx, options.eps)?;

        // Compute J^T * J using numr's transpose
        let jt = jacobian
            .transpose(0, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: transpose - {}", e),
            })?;
        let jtj = client
            .matmul(&jt, &jacobian)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: J^T J - {}", e),
            })?;

        // Add lambda * I to J^T * J
        let jtj_damped = add_lambda_identity(client, &jtj, lambda)?;

        // Compute J^T * f
        let fx_col = fx
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: reshape fx - {}", e),
            })?;
        let jtf = client
            .matmul(&jt, &fx_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: J^T f - {}", e),
            })?;

        // Solve (J^T J + lambda I) * dx = -J^T * f using numr's solve
        let neg_jtf = client
            .mul_scalar(&jtf, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: negate jtf - {}", e),
            })?;

        let dx_col = match TensorOps::solve(client, &jtj_damped, &neg_jtf) {
            Ok(dx) => dx,
            Err(_) => {
                lambda *= lambda_up;
                lambda = lambda.clamp(lambda_min, lambda_max);
                continue;
            }
        };

        let dx = dx_col
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: reshape dx - {}", e),
            })?;

        // x_new = x + dx
        let x_new = client
            .add(&x, &dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: update x - {}", e),
            })?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("levenberg_marquardt: evaluation - {}", e),
        })?;

        let new_res_norm =
            tensor_norm(client, &fx_new).map_err(|e| OptimizeError::NumericalError {
                message: format!("levenberg_marquardt: new norm - {}", e),
            })?;

        if new_res_norm < res_norm {
            // Accept step
            x = x_new;
            fx = fx_new;
            lambda *= lambda_down;

            let dx_norm =
                tensor_norm(client, &dx).map_err(|e| OptimizeError::NumericalError {
                    message: format!("levenberg_marquardt: dx norm - {}", e),
                })?;

            if dx_norm < options.x_tol {
                return Ok(TensorRootResult {
                    x,
                    fun: fx,
                    iterations: iter + 1,
                    residual_norm: new_res_norm,
                    converged: true,
                });
            }
        } else {
            // Reject step, increase damping
            lambda *= lambda_up;
        }

        lambda = lambda.clamp(lambda_min, lambda_max);
    }

    let final_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
        message: format!("levenberg_marquardt: final norm - {}", e),
    })?;

    Ok(TensorRootResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        residual_norm: final_norm,
        converged: false,
    })
}

/// Add lambda * I to matrix A using numr's diagflat.
fn add_lambda_identity<R, C>(client: &C, a: &Tensor<R>, lambda: f64) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n = a.shape()[0];

    // Create vector of ones, then scale by lambda, then diagflat
    let ones_data = vec![lambda; n];
    let lambda_vec = Tensor::<R>::from_slice(&ones_data, &[n], client.device());

    // Use numr's diagflat to create lambda * I
    let lambda_i = TensorOps::diagflat(client, &lambda_vec)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_lambda_identity: diagflat - {}", e),
        })?;

    client
        .add(a, &lambda_i)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_lambda_identity: add - {}", e),
        })
}
