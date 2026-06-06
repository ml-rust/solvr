//! Newton's method for systems of nonlinear equations using tensor operations.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
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
    R: Runtime<DType = DType>,
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

        let dx_col =
            LinearAlgebraAlgorithms::solve(client, &jacobian, &neg_fx_col).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("newton_system: solve - {}", e),
                }
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
/// All operations stay on device - no to_vec()/from_slice().
pub fn finite_difference_jacobian_tensor<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: &Tensor<R>,
    eps: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x.shape()[0];

    // Create identity matrix scaled by eps - each column is eps * e_j
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: eye - {}", e),
        })?;
    let eps_identity =
        client
            .mul_scalar(&identity, eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: scale identity - {}", e),
            })?;

    // Compute each column of the Jacobian
    let mut jac_columns: Vec<Tensor<R>> = Vec::with_capacity(n);

    for j in 0..n {
        // Extract column j as delta vector (transpose row j)
        let delta = eps_identity
            .narrow(0, j, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: narrow row - {}", e),
            })?
            .contiguous()
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: contiguous delta - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: reshape delta - {}", e),
            })?;

        // x_plus = x + delta
        let x_plus = client
            .add(x, &delta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: x + delta - {}", e),
            })?;

        // f(x_plus)
        let fx_plus = f(&x_plus).map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: f(x+delta) - {}", e),
        })?;

        // jac_col = (fx_plus - fx) / eps
        let diff = client
            .sub(&fx_plus, fx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: fx_plus - fx - {}", e),
            })?;
        let jac_col =
            client
                .mul_scalar(&diff, 1.0 / eps)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("jacobian: scale diff - {}", e),
                })?;

        // Reshape to [n, 1] for stacking
        let jac_col_2d = jac_col
            .unsqueeze(1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: unsqueeze col - {}", e),
            })?;
        jac_columns.push(jac_col_2d);
    }

    // Concatenate columns: [n, 1] * n -> [n, n]
    let refs: Vec<&Tensor<R>> = jac_columns.iter().collect();
    client
        .cat(&refs, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: cat columns - {}", e),
        })
}
