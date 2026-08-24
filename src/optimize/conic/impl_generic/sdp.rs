//! SDP interior point solver.
//!
//! Uses a simplified barrier method for semidefinite programming:
//!   min  trace(C*X)
//!   s.t. trace(A_i*X) = b_i
//!        X >= 0  (positive semidefinite)

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::conic::traits::sdp::{SdpOptions, SdpResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};

/// SDP interior point implementation.
///
/// Solves min trace(C*X) s.t. trace(A_i*X) = b_i, X >= 0 (PSD).
/// Uses a simplified barrier method with gradient descent.
pub fn sdp_impl<R, C>(
    client: &C,
    c_mat: &Tensor<R>,
    a_mats: &[Tensor<R>],
    b_vec: &Tensor<R>,
    options: &SdpOptions,
) -> OptimizeResult<SdpResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let shape = c_mat.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(OptimizeError::InvalidInput {
            context: "sdp: C must be square matrix".to_string(),
        });
    }

    let n = shape[0];
    let m = a_mats.len();

    if m == 0 {
        // Unconstrained: optimal is 0 matrix (with trace = 0)
        let x =
            client
                .fill(&[n, n], 0.0, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("sdp: fill zero - {}", e),
                })?;
        let fun = 0.0;
        let dual =
            client
                .fill(&[m], 0.0, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("sdp: fill dual - {}", e),
                })?;
        return Ok(SdpResult {
            x,
            fun,
            dual,
            iterations: 0,
            converged: true,
        });
    }

    // Initialize X = identity (strictly feasible starting point)
    let mut x = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sdp: eye - {}", e),
        })?;

    // Initialize dual variables
    let mut y = client
        .fill(&[m], 0.0, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sdp: fill y - {}", e),
        })?;

    let mut mu = 1.0;
    let mu_factor = 0.5;

    for iter in 0..options.max_iter {
        // Check convergence: barrier parameter small enough
        if mu < options.tol {
            let fun = compute_objective(client, c_mat, &x).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("sdp: objective - {}", e),
                }
            })?;
            return Ok(SdpResult {
                x,
                fun,
                dual: y,
                iterations: iter + 1,
                converged: true,
            });
        }

        // Compute primal residuals: r_i = trace(A_i*X) - b_i
        let mut residuals = Vec::with_capacity(m);
        for (i, a_i) in a_mats.iter().enumerate() {
            let trace_aix = compute_trace_product(client, a_i, &x).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("sdp: trace A_i*X iteration {} - {}", i, e),
                }
            })?;
            residuals.push(trace_aix);
        }

        // Convert residuals to tensor and subtract b_vec
        let residuals_tensor = Tensor::<R>::from_slice(&residuals, &[m], client.device())?;
        let residuals_tensor =
            client
                .sub(&residuals_tensor, b_vec)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("sdp: compute residuals - {}", e),
                })?;

        // Get max residual for step size (acceptable single scalar for control flow)
        let residuals_abs =
            client
                .abs(&residuals_tensor)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("sdp: abs residuals - {}", e),
                })?;
        let max_residual_tensor =
            client
                .max(&residuals_abs, &[], false)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("sdp: max residual - {}", e),
                })?;
        let max_residual: f64 = max_residual_tensor.to_vec()[0];

        // Update dual variables: y_new = y - 0.05 * residuals
        let scaled_residuals = client.mul_scalar(&residuals_tensor, -0.05).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("sdp: scale residuals - {}", e),
            }
        })?;

        y = client
            .add(&y, &scaled_residuals)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("sdp: update y - {}", e),
            })?;

        // Compute primal gradient: grad_X = C - sum_i y_i * A_i
        // For each constraint i, scale A_i by y_i and accumulate
        // Note: y_vals extraction happens once after convergence decision, then in gradient computation
        let y_vals: Vec<f64> = y.to_vec();
        let mut grad_x = c_mat.clone();

        for (i, a_i) in a_mats.iter().enumerate() {
            let weighted_a =
                client
                    .mul_scalar(a_i, y_vals[i])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("sdp: scale A[{}] - {}", i, e),
                    })?;

            grad_x =
                client
                    .sub(&grad_x, &weighted_a)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("sdp: grad_x update - {}", e),
                    })?;
        }

        // Gradient step on X
        let step_size = mu / (max_residual.max(1e-4) + 1e-10);

        let direction = client.mul_scalar(&grad_x, -0.01 * step_size).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("sdp: direction - {}", e),
            }
        })?;

        x = client
            .add(&x, &direction)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("sdp: x update - {}", e),
            })?;

        mu *= mu_factor;
    }

    let fun = compute_objective(client, c_mat, &x).map_err(|e| OptimizeError::NumericalError {
        message: format!("sdp: final objective - {}", e),
    })?;

    Ok(SdpResult {
        x,
        fun,
        dual: y,
        iterations: options.max_iter,
        converged: false,
    })
}

/// Compute trace(A*X) where both A and X are matrices.
fn compute_trace_product<R, C>(client: &C, a: &Tensor<R>, x: &Tensor<R>) -> Result<f64, String>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let product = client.matmul(a, x).map_err(|e| e.to_string())?;
    trace_matrix(client, &product)
}

/// Compute trace of a matrix.
fn trace_matrix<R, C>(client: &C, matrix: &Tensor<R>) -> Result<f64, String>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err("trace: input must be square matrix".to_string());
    }

    let trace_tensor = client.trace(matrix).map_err(|e| e.to_string())?;
    // trace returns a 1D tensor with single element, extract it
    let trace_vals = trace_tensor.to_vec();
    Ok(trace_vals[0])
}

/// Compute objective: trace(C*X).
fn compute_objective<R, C>(client: &C, c: &Tensor<R>, x: &Tensor<R>) -> Result<f64, String>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    compute_trace_product(client, c, x)
}
