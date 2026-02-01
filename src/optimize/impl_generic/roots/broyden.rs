//! Broyden's method for systems of nonlinear equations using tensor operations.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::roots::RootOptions;

use super::TensorRootResult;
use super::newton::finite_difference_jacobian_tensor;
use crate::optimize::impl_generic::utils::{tensor_norm, tensor_dot, SINGULAR_THRESHOLD};

/// Broyden's method (rank-1 update) for systems of nonlinear equations.
///
/// A quasi-Newton method that approximates the Jacobian using rank-1 updates.
pub fn broyden1_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &RootOptions,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "broyden1: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("broyden1: initial evaluation - {}", e),
    })?;

    if fx.shape()[0] != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "broyden1: function returns {} values but input has {} dimensions",
                fx.shape()[0],
                n
            ),
        });
    }

    // Initialize Jacobian approximation
    let mut jacobian = finite_difference_jacobian_tensor(client, &f, &x, &fx, options.eps)?;

    for iter in 0..options.max_iter {
        let res_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
            message: format!("broyden1: norm - {}", e),
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

        // Solve J * dx = -fx
        let neg_fx = client
            .mul_scalar(&fx, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: negate fx - {}", e),
            })?;

        // Reshape for solve: b needs to be [n, 1] for matrix solve
        let neg_fx_col = neg_fx
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: reshape neg_fx - {}", e),
            })?;

        let dx = match TensorOps::solve(client, &jacobian, &neg_fx_col) {
            Ok(dx_col) => dx_col
                .reshape(&[n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("broyden1: reshape dx - {}", e),
                })?,
            Err(_) => {
                // Reset Jacobian if singular
                jacobian = finite_difference_jacobian_tensor(client, &f, &x, &fx, options.eps)?;
                match TensorOps::solve(client, &jacobian, &neg_fx_col) {
                    Ok(dx_col) => dx_col
                        .reshape(&[n])
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("broyden1: reshape dx - {}", e),
                        })?,
                    Err(e) => {
                        return Err(OptimizeError::NumericalError {
                            message: format!("broyden1: solve - {}", e),
                        })
                    }
                }
            }
        };

        // x_new = x + dx
        let x_new = client
            .add(&x, &dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: update x - {}", e),
            })?;

        let dx_norm = tensor_norm(client, &dx).map_err(|e| OptimizeError::NumericalError {
            message: format!("broyden1: dx norm - {}", e),
        })?;

        if dx_norm < options.x_tol {
            let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: final evaluation - {}", e),
            })?;
            let final_norm =
                tensor_norm(client, &fx_new).map_err(|e| OptimizeError::NumericalError {
                    message: format!("broyden1: final norm - {}", e),
                })?;
            return Ok(TensorRootResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                residual_norm: final_norm,
                converged: true,
            });
        }

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("broyden1: evaluation - {}", e),
        })?;

        // Broyden rank-1 update: J_new = J + (df - J*dx) * dx^T / (dx^T * dx)
        // df = fx_new - fx
        let df = client
            .sub(&fx_new, &fx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: df = fx_new - fx - {}", e),
            })?;

        // J * dx
        let j_dx = matmul_vector(client, &jacobian, &dx)?;

        // diff = df - J*dx
        let diff = client
            .sub(&df, &j_dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("broyden1: diff - {}", e),
            })?;

        // dx^T * dx
        let dx_dot_dx = tensor_dot(client, &dx, &dx).map_err(|e| OptimizeError::NumericalError {
            message: format!("broyden1: dx dot dx - {}", e),
        })?;

        if dx_dot_dx > SINGULAR_THRESHOLD {
            // Update: J = J + diff * dx^T / dx_dot_dx
            jacobian = update_jacobian_rank1(client, &jacobian, &diff, &dx, dx_dot_dx)?;
        }

        x = x_new;
        fx = fx_new;
    }

    let final_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
        message: format!("broyden1: final norm - {}", e),
    })?;

    Ok(TensorRootResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        residual_norm: final_norm,
        converged: false,
    })
}

/// Compute matrix-vector product: A * x where A is [n,n] and x is [n].
fn matmul_vector<R, C>(client: &C, a: &Tensor<R>, x: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];

    // Reshape x to [n, 1] for matmul
    let x_col = x
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("matmul_vector: reshape x - {}", e),
        })?;

    let result = client
        .matmul(a, &x_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("matmul_vector: matmul - {}", e),
        })?;

    // Reshape back to [n]
    result
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("matmul_vector: reshape result - {}", e),
        })
}

/// Update Jacobian with rank-1 update: J = J + u * v^T / c
fn update_jacobian_rank1<R, C>(
    client: &C,
    j: &Tensor<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
    c: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n = u.shape()[0];

    // u_col = u reshaped to [n, 1]
    let u_col = u
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("rank1: reshape u - {}", e),
        })?;

    // v_row = v reshaped to [1, n]
    let v_row = v
        .reshape(&[1, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("rank1: reshape v - {}", e),
        })?;

    // outer = u * v^T  [n, 1] @ [1, n] = [n, n]
    let outer = client
        .matmul(&u_col, &v_row)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("rank1: outer product - {}", e),
        })?;

    // scaled = outer / c
    let scaled = client
        .mul_scalar(&outer, 1.0 / c)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("rank1: scale - {}", e),
        })?;

    // J_new = J + scaled
    client
        .add(j, &scaled)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("rank1: add - {}", e),
        })
}

