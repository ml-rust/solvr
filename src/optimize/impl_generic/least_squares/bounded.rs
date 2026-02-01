//! Bounded Levenberg-Marquardt algorithm for nonlinear least squares using tensors.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::utils::{
    compute_cost as utils_compute_cost, finite_difference_jacobian as utils_finite_difference_jacobian,
    tensor_norm as utils_tensor_norm, SINGULAR_THRESHOLD,
};
use crate::optimize::least_squares::LeastSquaresOptions;

use super::leastsq::leastsq_impl;
use super::TensorLeastSquaresResult;

/// Bounded Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// Minimizes ||f(x)||^2 where f: R^n -> R^m, subject to bounds constraints.
pub fn least_squares_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    bounds: Option<(&Tensor<R>, &Tensor<R>)>,
    options: &LeastSquaresOptions,
) -> OptimizeResult<TensorLeastSquaresResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: empty initial guess".to_string(),
        });
    }

    // If no bounds, use unbounded algorithm
    if bounds.is_none() {
        return leastsq_impl(client, f, x0, options);
    }

    let (lower, upper) = bounds.unwrap();

    if lower.shape()[0] != n || upper.shape()[0] != n {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: bounds dimension mismatch".to_string(),
        });
    }

    // Project initial guess onto bounds using numr ops
    let mut x = project_onto_bounds(client, x0, lower, upper)?;

    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("least_squares: initial evaluation - {}", e),
    })?;

    let m = fx.shape()[0];
    if m == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: residual function returns empty vector".to_string(),
        });
    }

    let mut nfev = 1;
    let mut cost = compute_cost(client, &fx)?;

    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let lambda_min = SINGULAR_THRESHOLD;
    let lambda_max = 1e10;

    for iter in 0..options.max_iter {
        if cost < options.f_tol {
            return Ok(TensorLeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute Jacobian using finite differences
        let jacobian = finite_difference_jacobian(client, &f, &x, &fx, m, n, options.eps)?;
        nfev += n;

        // Compute J^T J using numr's transpose
        let jt = jacobian
            .transpose(0, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: transpose - {}", e),
            })?;
        let jtj = client
            .matmul(&jt, &jacobian)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: J^T J - {}", e),
            })?;

        // Add lambda * diag(J^T J) damping
        let jtj_damped = add_scaled_diagonal(client, &jtj, lambda, n)?;

        // Compute J^T f
        let fx_col = fx
            .reshape(&[m, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: reshape fx - {}", e),
            })?;
        let jtf = client
            .matmul(&jt, &fx_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: J^T f - {}", e),
            })?;
        let jtf_vec = jtf
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: reshape jtf - {}", e),
            })?;

        // Check gradient norm for convergence
        let grad_norm = tensor_norm(client, &jtf_vec)?;
        if grad_norm < options.g_tol {
            return Ok(TensorLeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Solve (J^T J + lambda*diag) * dx = -J^T f using numr's solve
        let neg_jtf = client
            .mul_scalar(&jtf, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: negate jtf - {}", e),
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
                message: format!("least_squares: reshape dx - {}", e),
            })?;

        // x_new = x + dx, projected onto bounds
        let x_unbounded = client
            .add(&x, &dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("least_squares: update x - {}", e),
            })?;
        let x_new = project_onto_bounds(client, &x_unbounded, lower, upper)?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("least_squares: evaluation - {}", e),
        })?;
        nfev += 1;

        let cost_new = compute_cost(client, &fx_new)?;

        if cost_new < cost {
            // Accept step - compute actual step taken (after projection)
            let actual_dx = client
                .sub(&x_new, &x)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("least_squares: actual dx - {}", e),
                })?;
            let dx_norm = tensor_norm(client, &actual_dx)?;

            if dx_norm < options.x_tol {
                return Ok(TensorLeastSquaresResult {
                    x: x_new,
                    residuals: fx_new,
                    cost: cost_new,
                    iterations: iter + 1,
                    nfev,
                    converged: true,
                });
            }

            x = x_new;
            fx = fx_new;
            cost = cost_new;
            lambda *= lambda_down;
        } else {
            // Reject step, increase damping
            lambda *= lambda_up;
        }

        lambda = lambda.clamp(lambda_min, lambda_max);
    }

    Ok(TensorLeastSquaresResult {
        x,
        residuals: fx,
        cost,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Project tensor onto bounds using numr's minimum/maximum.
fn project_onto_bounds<R, C>(
    client: &C,
    x: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // clamp(x, lower, upper) = min(max(x, lower), upper)
    let clamped_low = client
        .maximum(x, lower)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("project_onto_bounds: max - {}", e),
        })?;
    client
        .minimum(&clamped_low, upper)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("project_onto_bounds: min - {}", e),
        })
}

/// Wrappers for shared utility functions with OptimizeError mapping.
fn compute_cost<R, C>(client: &C, fx: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    utils_compute_cost(client, fx).map_err(|e| OptimizeError::NumericalError {
        message: format!("compute_cost: {}", e),
    })
}

fn tensor_norm<R, C>(client: &C, v: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    utils_tensor_norm(client, v).map_err(|e| OptimizeError::NumericalError {
        message: format!("tensor_norm: {}", e),
    })
}

fn finite_difference_jacobian<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: &Tensor<R>,
    m: usize,
    n: usize,
    eps: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    utils_finite_difference_jacobian(client, f, x, fx, m, n, eps).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("finite_difference_jacobian: {}", e),
        }
    })
}

/// Add lambda * max(|diag(A)|, threshold) to diagonal of A.
/// Uses tensor ops throughout - no to_vec()/from_slice().
fn add_scaled_diagonal<R, C>(
    client: &C,
    a: &Tensor<R>,
    lambda: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    // Extract diagonal using numr's diag
    let diag_vec = TensorOps::diag(client, a)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: diag - {}", e),
        })?;

    // Compute abs(diag)
    let abs_diag = client
        .abs(&diag_vec)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: abs - {}", e),
        })?;

    // Create threshold tensor
    let threshold = client
        .fill(&[n], SINGULAR_THRESHOLD, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: threshold - {}", e),
        })?;

    // max(|diag|, threshold)
    let clamped_diag = client
        .maximum(&abs_diag, &threshold)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: max - {}", e),
        })?;

    // Scale by lambda
    let scaled_diag = client
        .mul_scalar(&clamped_diag, lambda)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: scale - {}", e),
        })?;

    // Create diagonal matrix using numr's diagflat
    let diag_matrix = TensorOps::diagflat(client, &scaled_diag)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: diagflat - {}", e),
        })?;

    client
        .add(a, &diag_matrix)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: add - {}", e),
        })
}
