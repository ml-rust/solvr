//! Shared helper functions for multivariate minimization.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};

use super::utils::tensor_dot;

// Re-export TensorMinimizeResult from traits for backwards compatibility
pub use crate::optimize::minimize::traits::TensorMinimizeResult;

/// Backtracking line search with Armijo condition using tensor operations.
///
/// All operations stay on device - no GPU->CPU transfers in the loop.
pub fn backtracking_line_search_tensor<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    p: &Tensor<R>,
    fx: f64,
    grad: &Tensor<R>,
) -> OptimizeResult<(Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let c = 0.0001;
    let rho = 0.5;

    let grad_dot_p = tensor_dot(client, grad, p).map_err(|e| OptimizeError::NumericalError {
        message: format!("line_search: grad_dot_p - {}", e),
    })?;

    let mut alpha = 1.0;
    let mut nfev = 0;

    for _ in 0..50 {
        // x_new = x + alpha * p (all on device)
        let scaled_p = client
            .mul_scalar(p, alpha)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("line_search: scale p - {}", e),
            })?;
        let x_new = client
            .add(x, &scaled_p)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("line_search: x + alpha*p - {}", e),
            })?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("line_search: f eval - {}", e),
        })?;
        nfev += 1;

        if fx_new <= fx + c * alpha * grad_dot_p {
            return Ok((x_new, fx_new, nfev));
        }

        alpha *= rho;
    }

    Ok((x.clone(), fx, nfev))
}

/// Line search for Powell's method using tensor operations.
///
/// Searches along the given direction to find a point with lower function value.
/// All operations stay on device - no GPU->CPU transfers in the loop.
pub fn line_search_tensor<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    direction: &Tensor<R>,
    fx: f64,
) -> OptimizeResult<(Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut alpha = 0.1;
    let mut nfev = 0;

    let mut best_x = x.clone();
    let mut best_fx = fx;

    for _ in 0..20 {
        // x_new = x + alpha * direction (all on device)
        let scaled_dir =
            client
                .mul_scalar(direction, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell line_search: scale - {}", e),
                })?;
        let x_new = client
            .add(x, &scaled_dir)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell line_search: add - {}", e),
            })?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("powell line_search: f eval - {}", e),
        })?;
        nfev += 1;

        if fx_new < best_fx {
            best_x = x_new;
            best_fx = fx_new;
            alpha *= 1.5;
        } else {
            alpha *= 0.5;
            if alpha < 1e-10 {
                break;
            }
        }
    }

    Ok((best_x, best_fx, nfev))
}

/// Compare two f64 values, treating NaN as greater than all other values.
/// This ensures NaN values sort to the end.
pub fn compare_f64_nan_safe(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // If comparison fails (one or both is NaN):
        // - NaN should sort to the end (be "greater")
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => std::cmp::Ordering::Equal, // unreachable
        }
    })
}

// ============================================================================
// Hessian-Vector Product (HVP) Utilities
// ============================================================================
//
// Hessian-Vector Product (HVP) utilities for second-order optimization.
//
// This section provides functions for computing Hessian-vector products without
// forming the full Hessian matrix. For a scalar function f: ℝⁿ → ℝ, the HVP
// computes H(x) @ v where H is the n×n Hessian matrix.
//
// # Memory Efficiency
//
// Computing the full Hessian requires O(n²) memory, which is prohibitive for
// large-scale optimization. HVP requires only O(n) memory per product.
//
// # Two Approaches
//
// 1. **Reverse-over-reverse (double backward)**: Uses `backward_with_graph` + `backward`
//    - More memory efficient for scalar loss functions
//    - Standard approach for neural network training
//
// 2. **Forward-over-reverse**: Uses `jvp` through the gradient computation
//    - Available via `numr::autograd::hvp`
//    - May be more efficient when gradient function is already available

use numr::autograd::{Var, VarGradStore, backward, backward_with_graph, var_mul, var_sum};

/// Compute Hessian-vector product using reverse-over-reverse (double backward).
///
/// For a scalar function f: ℝⁿ → ℝ represented by the loss `Var`, computes H @ v
/// where H is the Hessian matrix ∂²f/∂x².
///
/// # Algorithm
///
/// 1. Compute gradient ∇f(x) using `backward_with_graph` (preserves computation graph)
/// 2. Compute scalar product g = ∇f(x) · v
/// 3. Compute gradient of g w.r.t. x using `backward` → gives H @ v
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `loss` - The scalar loss Var (output of differentiable computation)
/// * `x` - The input Var to differentiate with respect to
/// * `v` - The vector to multiply with the Hessian
///
/// # Returns
///
/// The Hessian-vector product H(x) @ v as a Tensor.
///
/// # Example
///
/// ```ignore
/// // f(x) = x², H = 2, so H @ v = 2v
/// let x = Var::new(tensor, true);
/// let loss = var_mul(&x, &x, &client)?;
/// let v = Tensor::ones(&[n], dtype, device);
/// let hvp = hvp_reverse_over_reverse(&client, &loss, &x, &v)?;
/// ```
pub fn hvp_reverse_over_reverse<R, C>(
    client: &C,
    loss: &Var<R>,
    x: &Var<R>,
    v: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    // Step 1: First backward pass - get gradient as Var (preserves graph)
    let grads: VarGradStore<R> =
        backward_with_graph(loss, client).map_err(|e| OptimizeError::NumericalError {
            message: format!("hvp: backward_with_graph failed - {}", e),
        })?;

    // Get gradient as Var (NOT Tensor!) - this retains the computation graph
    let grad_x: &Var<R> = grads
        .get_var(x.id())
        .ok_or_else(|| OptimizeError::NumericalError {
            message: "hvp: no gradient for x (is x a leaf with requires_grad=true?)".to_string(),
        })?;

    // Step 2: Compute dot product ∇f · v
    // v is a constant (no gradient needed)
    let v_var = Var::new(v.clone(), false);

    // Element-wise multiplication: grad_x * v
    let grad_v: Var<R> =
        var_mul(grad_x, &v_var, client).map_err(|e| OptimizeError::NumericalError {
            message: format!("hvp: var_mul failed - {}", e),
        })?;

    // Sum to scalar: sum(grad_x * v) = ∇f · v
    let grad_v_dot: Var<R> =
        var_sum(&grad_v, &[0], false, client).map_err(|e| OptimizeError::NumericalError {
            message: format!("hvp: var_sum failed - {}", e),
        })?;

    // Step 3: Second backward pass - differentiate (∇f · v) w.r.t. x
    // This gives ∂/∂x(∇f · v) = H @ v
    let hvp_grads = backward(&grad_v_dot, client).map_err(|e| OptimizeError::NumericalError {
        message: format!("hvp: second backward failed - {}", e),
    })?;

    hvp_grads
        .get(x.id())
        .cloned()
        .ok_or_else(|| OptimizeError::NumericalError {
            message: "hvp: no HVP gradient for x".to_string(),
        })
}

/// Compute Hessian-vector product for a function given as a closure.
///
/// This is a convenience wrapper that creates the Var, evaluates the function,
/// and computes the HVP in one call.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function that takes a Var and returns a scalar Var (the loss)
/// * `x` - Point at which to evaluate (as Tensor)
/// * `v` - Direction vector for HVP
///
/// # Returns
///
/// Tuple of (f(x), H(x) @ v) - both the function value and HVP.
///
/// # Scalar Extraction Note
///
/// This function uses `tensor.item()` to extract the scalar loss value. This is
/// necessary because scalar loss values must be available as f64 for control flow
/// (convergence checks, line search). For single-element tensors, this transfer
/// is minimal (~8 bytes) and unavoidable in second-order optimization algorithms.
pub fn hvp_from_fn<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> OptimizeResult<(f64, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> Result<Var<R>>,
{
    // Create input Var with gradient tracking
    let x_var = Var::new(x.clone(), true);

    // Evaluate function
    let loss = f(&x_var, client).map_err(|e| OptimizeError::NumericalError {
        message: format!("hvp_from_fn: function evaluation failed - {}", e),
    })?;

    // Extract scalar value (single-element tensor → f64)
    // NOTE: This is an unavoidable device→CPU transfer for the final loss value.
    // The transfer is minimal (8 bytes) and only happens once per HVP computation.
    let loss_value: f64 =
        loss.tensor()
            .item::<f64>()
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("hvp_from_fn: scalar extraction - {}", e),
            })?;

    // Compute HVP
    let hvp = hvp_reverse_over_reverse(client, &loss, &x_var, v)?;

    Ok((loss_value, hvp))
}

/// Compute gradient using autograd (reverse-mode).
///
/// For a scalar function f: ℝⁿ → ℝ, computes ∇f(x).
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function that takes a Var and returns a scalar Var
/// * `x` - Point at which to evaluate
///
/// # Returns
///
/// Tuple of (f(x), ∇f(x)) - both the function value and gradient.
///
/// # Scalar Extraction Note
///
/// This function uses `tensor.item()` to extract the scalar loss value. See `hvp_from_fn`
/// documentation for rationale.
pub fn gradient_from_fn<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
) -> OptimizeResult<(f64, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> Result<Var<R>>,
{
    // Create input Var with gradient tracking
    let x_var = Var::new(x.clone(), true);

    // Evaluate function
    let loss = f(&x_var, client).map_err(|e| OptimizeError::NumericalError {
        message: format!("gradient_from_fn: function evaluation failed - {}", e),
    })?;

    // Extract scalar value (single-element tensor → f64)
    // NOTE: Minimal transfer (8 bytes), unavoidable for returning f64 to caller.
    let loss_value: f64 =
        loss.tensor()
            .item::<f64>()
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("gradient_from_fn: scalar extraction - {}", e),
            })?;

    // Compute gradient
    let grads = backward(&loss, client).map_err(|e| OptimizeError::NumericalError {
        message: format!("gradient_from_fn: backward failed - {}", e),
    })?;

    let grad = grads
        .get(x_var.id())
        .cloned()
        .ok_or_else(|| OptimizeError::NumericalError {
            message: "gradient_from_fn: no gradient for x".to_string(),
        })?;

    Ok((loss_value, grad))
}

#[cfg(test)]
mod hvp_tests {
    use super::*;
    use numr::autograd::var_sum;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_hvp_quadratic() {
        let (device, client) = setup();

        // f(x) = sum(x²) = x₁² + x₂² + x₃²
        // ∇f = [2x₁, 2x₂, 2x₃]
        // H = diag([2, 2, 2])
        // H @ v = [2v₁, 2v₂, 2v₃]
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let (fx, hvp) = hvp_from_fn(
            &client,
            |x_var, c| {
                let x_sq = var_mul(x_var, x_var, c)?;
                var_sum(&x_sq, &[0], false, c) // Sum over dim 0 to get scalar
            },
            &x,
            &v,
        )
        .unwrap();

        // f(x) = 1 + 4 + 9 = 14
        assert!((fx - 14.0).abs() < 1e-10);

        // H @ v = [2, 2, 2]
        let hvp_vals: Vec<f64> = hvp.to_vec();
        assert!((hvp_vals[0] - 2.0).abs() < 1e-10);
        assert!((hvp_vals[1] - 2.0).abs() < 1e-10);
        assert!((hvp_vals[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hvp_with_direction() {
        let (device, client) = setup();

        // f(x) = sum(x²)
        // H @ v where v = [1, 0, 0] should give [2, 0, 0]
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0], &[3], &device);

        let (_, hvp) = hvp_from_fn(
            &client,
            |x_var, c| {
                let x_sq = var_mul(x_var, x_var, c)?;
                var_sum(&x_sq, &[0], false, c) // Sum over dim 0 to get scalar
            },
            &x,
            &v,
        )
        .unwrap();

        let hvp_vals: Vec<f64> = hvp.to_vec();
        assert!((hvp_vals[0] - 2.0).abs() < 1e-10);
        assert!((hvp_vals[1] - 0.0).abs() < 1e-10);
        assert!((hvp_vals[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_from_fn() {
        let (device, client) = setup();

        // f(x) = sum(x²)
        // ∇f = 2x
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let (fx, grad) = gradient_from_fn(
            &client,
            |x_var, c| {
                let x_sq = var_mul(x_var, x_var, c)?;
                var_sum(&x_sq, &[0], false, c) // Sum over dim 0 to get scalar
            },
            &x,
        )
        .unwrap();

        assert!((fx - 14.0).abs() < 1e-10);

        let grad_vals: Vec<f64> = grad.to_vec();
        assert!((grad_vals[0] - 2.0).abs() < 1e-10);
        assert!((grad_vals[1] - 4.0).abs() < 1e-10);
        assert!((grad_vals[2] - 6.0).abs() < 1e-10);
    }
}
