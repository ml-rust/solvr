//! Newton's method for systems of nonlinear equations using tensor operations.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::roots::RootOptions;

use super::TensorRootResult;
use crate::optimize::impl_generic::utils::tensor_norm;

/// Newton's method for systems of nonlinear equations using tensors.
///
/// Finds x such that F(x) ≈ 0 where F: R^n -> R^n.
pub fn newton_system_impl<R, C, F>(
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
            context: "newton_system: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("newton_system: initial evaluation - {}", e),
    })?;

    if fx.shape()[0] != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "newton_system: function returns {} values but input has {} dimensions",
                fx.shape()[0],
                n
            ),
        });
    }

    for iter in 0..options.max_iter {
        let res_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
            message: format!("newton_system: norm - {}", e),
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

        // Compute Jacobian using finite differences
        let jacobian = finite_difference_jacobian_tensor(client, &f, &x, &fx, options.eps)?;

        // Solve J * dx = -fx using numr's solve
        let neg_fx = client
            .mul_scalar(&fx, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: negate fx - {}", e),
            })?;

        // Reshape for solve: b needs to be [n, 1] for matrix solve
        let neg_fx_col = neg_fx
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: reshape neg_fx - {}", e),
            })?;

        let dx_col = TensorOps::solve(client, &jacobian, &neg_fx_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: solve - {}", e),
            })?;

        let dx = dx_col
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: reshape dx - {}", e),
            })?;

        // x = x + dx
        x = client
            .add(&x, &dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: update x - {}", e),
            })?;

        let dx_norm = tensor_norm(client, &dx).map_err(|e| OptimizeError::NumericalError {
            message: format!("newton_system: dx norm - {}", e),
        })?;

        if dx_norm < options.x_tol {
            fx = f(&x).map_err(|e| OptimizeError::NumericalError {
                message: format!("newton_system: final evaluation - {}", e),
            })?;
            let final_norm =
                tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
                    message: format!("newton_system: final norm - {}", e),
                })?;
            return Ok(TensorRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: final_norm,
                converged: true,
            });
        }

        fx = f(&x).map_err(|e| OptimizeError::NumericalError {
            message: format!("newton_system: evaluation - {}", e),
        })?;
    }

    let final_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
        message: format!("newton_system: final norm - {}", e),
    })?;

    Ok(TensorRootResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        residual_norm: final_norm,
        converged: false,
    })
}

/// Compute Jacobian matrix using finite differences.
/// Returns [n, n] tensor where J[i,j] = df_i/dx_j.
///
/// Note: This function necessarily uses to_vec() because we need to perturb
/// individual elements. This is acceptable since finite differences require
/// n function evaluations anyway, so the overhead is minimal.
pub fn finite_difference_jacobian_tensor<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: &Tensor<R>,
    eps: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x.shape()[0];

    // We need to perturb each element individually, which requires accessing values.
    // This is inherent to finite difference Jacobian computation.
    let x_data: Vec<f64> = x.to_vec();
    let fx_data: Vec<f64> = fx.to_vec();

    let mut jacobian_data = vec![0.0; n * n];

    for j in 0..n {
        // Create perturbed x: x + eps * e_j
        let mut x_plus_data = x_data.clone();
        x_plus_data[j] += eps;
        let x_plus = Tensor::<R>::from_slice(&x_plus_data, &[n], client.device());

        let fx_plus = f(&x_plus).map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: f(x+delta) - {}", e),
        })?;
        let fx_plus_data: Vec<f64> = fx_plus.to_vec();

        // J[i, j] = (f_i(x + eps*e_j) - f_i(x)) / eps
        for i in 0..n {
            jacobian_data[i * n + j] = (fx_plus_data[i] - fx_data[i]) / eps;
        }
    }

    Ok(Tensor::<R>::from_slice(&jacobian_data, &[n, n], client.device()))
}
