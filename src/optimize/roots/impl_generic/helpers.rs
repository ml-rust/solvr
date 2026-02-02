//! Shared helper functions for root-finding algorithms.

use numr::autograd::{DualTensor, jacobian_forward};
use numr::error::Result as NumrResult;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};

// ============================================================================
// Analytic Jacobian Computation
// ============================================================================
//
// Analytic Jacobian computation using autograd.
//
// This section provides utilities for computing exact Jacobians using automatic
// differentiation, as an alternative to finite differences.
//
// # Forward vs Reverse Mode for Jacobians
//
// For a function F: ℝⁿ → ℝᵐ, the Jacobian J is an m×n matrix.
//
// - **Forward-mode (JVP)**: Computes one column at a time (n passes for full Jacobian)
// - **Reverse-mode (VJP)**: Computes one row at a time (m passes for full Jacobian)
//
// Choose based on which is smaller: n (inputs) or m (outputs).
// For square systems (n = m), forward-mode is typically preferred as it's
// more memory-efficient.

/// Compute the Jacobian matrix using forward-mode AD.
///
/// For a function F: ℝⁿ → ℝᵐ, computes the m×n Jacobian matrix J where
/// J[i,j] = ∂Fᵢ/∂xⱼ.
///
/// This uses numr's forward-mode AD (`jacobian_forward`), which computes
/// n JVPs (one per input dimension).
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function that takes a `DualTensor` and returns a `DualTensor`.
///   Must use `dual_*` operations from numr's dual_ops module.
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
///
/// Jacobian matrix of shape [m, n] where m is output dimension and n is input dimension.
///
/// # Example
///
/// ```ignore
/// use numr::autograd::dual_ops::{dual_mul, dual_add};
///
/// // F(x) = [x₀², x₀ + x₁]
/// let jacobian = jacobian_forward_impl(
///     &client,
///     |x, c| {
///         // Build the vector output using dual operations
///         // ...
///     },
///     &x,
/// )?;
/// ```
pub fn jacobian_forward_impl<R, C, F>(client: &C, f: F, x: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &C) -> NumrResult<DualTensor<R>>,
{
    jacobian_forward(f, x, client).map_err(|e| OptimizeError::NumericalError {
        message: format!("jacobian_forward: {}", e),
    })
}

/// Compute Jacobian-vector product J @ v using forward-mode AD.
///
/// For a function F: ℝⁿ → ℝᵐ, computes J(x) @ v without forming the full Jacobian.
/// This is useful for iterative methods like Newton-Krylov.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function using dual operations
/// * `x` - Point at which to evaluate
/// * `v` - Vector to multiply with Jacobian
///
/// # Returns
///
/// Tuple of (F(x), J(x) @ v)
pub fn jvp_impl<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> OptimizeResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> NumrResult<DualTensor<R>>,
{
    numr::autograd::jvp(f, &[x], &[v], client).map_err(|e| OptimizeError::NumericalError {
        message: format!("jvp: {}", e),
    })
}

#[cfg(test)]
mod jacobian_tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul, dual_mul_scalar};
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_jacobian_linear() {
        let (device, client) = setup();

        // F(x) = 2x (linear function)
        // Jacobian = diag([2, 2, 2])
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let jacobian =
            jacobian_forward_impl(&client, |dual_x, c| dual_mul_scalar(dual_x, 2.0, c), &x)
                .unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Should be diagonal with 2s
        assert!((j[0] - 2.0).abs() < 1e-10); // [0,0]
        assert!((j[1] - 0.0).abs() < 1e-10); // [0,1]
        assert!((j[2] - 0.0).abs() < 1e-10); // [0,2]
        assert!((j[3] - 0.0).abs() < 1e-10); // [1,0]
        assert!((j[4] - 2.0).abs() < 1e-10); // [1,1]
        assert!((j[5] - 0.0).abs() < 1e-10); // [1,2]
        assert!((j[6] - 0.0).abs() < 1e-10); // [2,0]
        assert!((j[7] - 0.0).abs() < 1e-10); // [2,1]
        assert!((j[8] - 2.0).abs() < 1e-10); // [2,2]
    }

    #[test]
    fn test_jacobian_quadratic() {
        let (device, client) = setup();

        // F(x) = x² (element-wise)
        // Jacobian = diag([2x₀, 2x₁, 2x₂]) at x = [1, 2, 3]
        // = diag([2, 4, 6])
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let jacobian =
            jacobian_forward_impl(&client, |dual_x, c| dual_mul(dual_x, dual_x, c), &x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Should be diagonal with [2, 4, 6]
        assert!((j[0] - 2.0).abs() < 1e-10); // [0,0] = 2*1
        assert!((j[4] - 4.0).abs() < 1e-10); // [1,1] = 2*2
        assert!((j[8] - 6.0).abs() < 1e-10); // [2,2] = 2*3

        // Off-diagonals should be zero
        assert!((j[1]).abs() < 1e-10);
        assert!((j[2]).abs() < 1e-10);
        assert!((j[3]).abs() < 1e-10);
        assert!((j[5]).abs() < 1e-10);
        assert!((j[6]).abs() < 1e-10);
        assert!((j[7]).abs() < 1e-10);
    }
}
