//! L-BFGS (Limited-memory BFGS) quasi-Newton method for large-scale minimization.
//!
//! L-BFGS uses only O(mn) memory instead of O(n²) by storing m recent vector pairs
//! instead of the full inverse Hessian approximation.
//!
//! # When to Use L-BFGS vs BFGS
//!
//! **Use L-BFGS when:**
//! - Problem dimension n > 1000 (memory becomes prohibitive for BFGS)
//! - Memory is limited (embedded systems, browser via WebGPU)
//! - Training neural networks or fitting large models
//!
//! **Use BFGS when:**
//! - Problem dimension n < 100 (BFGS converges faster with full Hessian)
//! - Memory is abundant and performance is critical
//! - Function evaluations are expensive (fewer iterations matter)
//!
//! # Memory Comparison
//!
//! | Dimension (n) | BFGS Memory | L-BFGS Memory (m=10) | Reduction |
//! |---------------|-------------|----------------------|-----------|
//! | 100           | 80 KB       | 8 KB                 | 10x       |
//! | 1,000         | 8 MB        | 80 KB                | 100x      |
//! | 10,000        | 800 MB      | 800 KB               | 1000x     |
//! | 100,000       | 80 GB       | 8 MB                 | 10000x    |
//!
//! # Algorithm Details
//!
//! L-BFGS stores m recent correction pairs (s_k, y_k) where:
//! - s_k = x_{k+1} - x_k (position change)
//! - y_k = ∇f_{k+1} - ∇f_k (gradient change)
//!
//! The search direction is computed using the two-loop recursion algorithm,
//! which implicitly computes H_k @ ∇f without forming the matrix H_k.
//!
//! Typical values: m = 5-20 (default: 10)

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, backtracking_line_search_tensor};
use super::utils::{SINGULAR_THRESHOLD, finite_difference_gradient, tensor_dot, tensor_norm};

/// Options specific to L-BFGS.
#[derive(Debug, Clone)]
pub struct LbfgsOptions {
    /// Base minimization options
    pub base: MinimizeOptions,
    /// Number of correction pairs to store (default: 10)
    pub m: usize,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            base: MinimizeOptions::default(),
            m: 10,
        }
    }
}

/// L-BFGS quasi-Newton method for minimization using tensors.
///
/// Memory usage: O(mn) where m is history size (typically 10-20) and n is dimension.
/// All operations use tensor ops to stay on device (CPU/GPU).
pub fn lbfgs_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &LbfgsOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "lbfgs: empty initial guess".to_string(),
        });
    }

    if options.m == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "lbfgs: history size m must be > 0".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("lbfgs: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut grad =
        finite_difference_gradient(client, &f, &x, fx, options.base.eps).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("lbfgs: gradient - {}", e),
            }
        })?;
    nfev += n;

    // Storage for correction pairs (s_k, y_k)
    // s_k = x_{k+1} - x_k (position change)
    // y_k = g_{k+1} - g_k (gradient change)
    let mut s_history: Vec<Tensor<R>> = Vec::with_capacity(options.m);
    let mut y_history: Vec<Tensor<R>> = Vec::with_capacity(options.m);
    let mut rho_history: Vec<f64> = Vec::with_capacity(options.m);

    for iter in 0..options.base.max_iter {
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("lbfgs: grad norm - {}", e),
        })?;

        if grad_norm < options.base.g_tol {
            return Ok(TensorMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute search direction using two-loop recursion
        let p = two_loop_recursion(client, &grad, &s_history, &y_history, &rho_history)?;

        // Line search
        let (x_new, fx_new, evals) =
            backtracking_line_search_tensor(client, &f, &x, &p, fx, &grad)?;
        nfev += evals;

        // Compute step difference: s = x_new - x
        let s = client
            .sub(&x_new, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("lbfgs: s = x_new - x - {}", e),
            })?;

        // Check convergence based on step size
        let s_norm = tensor_norm(client, &s).map_err(|e| OptimizeError::NumericalError {
            message: format!("lbfgs: s norm - {}", e),
        })?;

        if s_norm < options.base.x_tol || (fx - fx_new).abs() < options.base.f_tol {
            return Ok(TensorMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient(client, &f, &x_new, fx_new, options.base.eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("lbfgs: new gradient - {}", e),
            })?;
        nfev += n;

        // Compute gradient difference: y = grad_new - grad
        let y = client
            .sub(&grad_new, &grad)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("lbfgs: y = grad_new - grad - {}", e),
            })?;

        // Compute rho = 1 / (y^T s)
        let ys = tensor_dot(client, &y, &s)?;

        if ys.abs() > SINGULAR_THRESHOLD {
            let rho = 1.0 / ys;

            // Update history (FIFO circular buffer)
            if s_history.len() >= options.m {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }

            s_history.push(s);
            y_history.push(y);
            rho_history.push(rho);
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    Ok(TensorMinimizeResult {
        x,
        fun: fx,
        iterations: options.base.max_iter,
        nfev,
        converged: false,
    })
}

/// Two-loop recursion for computing H_0 @ grad implicitly.
///
/// This computes the action of the approximate inverse Hessian on the gradient
/// without explicitly forming the matrix. Uses the stored correction pairs.
fn two_loop_recursion<R, C>(
    client: &C,
    grad: &Tensor<R>,
    s_history: &[Tensor<R>],
    y_history: &[Tensor<R>],
    rho_history: &[f64],
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    if s_history.is_empty() {
        // No history: use steepest descent direction -grad
        return client
            .mul_scalar(grad, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("two_loop: negate grad - {}", e),
            });
    }

    let m = s_history.len();
    let mut q = grad.clone();
    let mut alpha: Vec<f64> = vec![0.0; m];

    // First loop: backward through history
    for i in (0..m).rev() {
        // alpha_i = rho_i * s_i^T @ q
        let s_q = tensor_dot(client, &s_history[i], &q)?;
        alpha[i] = rho_history[i] * s_q;

        // q = q - alpha_i * y_i
        let alpha_y = client.mul_scalar(&y_history[i], alpha[i]).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("two_loop: alpha * y - {}", e),
            }
        })?;
        q = client
            .sub(&q, &alpha_y)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("two_loop: q - alpha*y - {}", e),
            })?;
    }

    // Initial inverse Hessian approximation: H_0 = (s^T y / y^T y) * I
    // We compute r = H_0 @ q
    let last_idx = m - 1;
    let s_y = tensor_dot(client, &s_history[last_idx], &y_history[last_idx])?;
    let y_y = tensor_dot(client, &y_history[last_idx], &y_history[last_idx])?;

    let gamma = if y_y.abs() > SINGULAR_THRESHOLD {
        s_y / y_y
    } else {
        1.0
    };

    let mut r = client
        .mul_scalar(&q, gamma)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("two_loop: H_0 @ q - {}", e),
        })?;

    // Second loop: forward through history
    for i in 0..m {
        // beta = rho_i * y_i^T @ r
        let y_r = tensor_dot(client, &y_history[i], &r)?;
        let beta = rho_history[i] * y_r;

        // r = r + s_i * (alpha_i - beta)
        let coeff = alpha[i] - beta;
        let coeff_s =
            client
                .mul_scalar(&s_history[i], coeff)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("two_loop: coeff * s - {}", e),
                })?;
        r = client
            .add(&r, &coeff_s)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("two_loop: r + coeff*s - {}", e),
            })?;
    }

    // Return search direction: -r (since we want to minimize)
    client
        .mul_scalar(&r, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("two_loop: negate result - {}", e),
        })
}
