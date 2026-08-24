//! Shared helper functions for root-finding algorithms.
use crate::DType;

use numr::autograd::DualTensor;
use numr::error::Result as NumrResult;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::common::jacobian::{jacobian_autograd, jvp_autograd};
use crate::optimize::error::{OptimizeError, OptimizeResult};

// ============================================================================
// Jacobian Wrappers with OptimizeError
// ============================================================================

/// Compute Jacobian using autograd with OptimizeError wrapping.
///
/// This is a thin wrapper around `common::jacobian_autograd` that converts
/// numr errors to OptimizeError for use in optimization algorithms.
pub fn jacobian_forward_impl<R, C, F>(client: &C, f: F, x: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &C) -> NumrResult<DualTensor<R>>,
{
    jacobian_autograd(client, f, x).map_err(|e| OptimizeError::NumericalError {
        message: format!("jacobian_autograd: {}", e),
    })
}

/// Compute Jacobian-vector product using autograd with OptimizeError wrapping.
pub fn jvp_impl<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> OptimizeResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> NumrResult<DualTensor<R>>,
{
    jvp_autograd(client, f, x, v).map_err(|e| OptimizeError::NumericalError {
        message: format!("jvp_autograd: {}", e),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul, dual_mul_scalar};
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
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let jacobian =
            jacobian_forward_impl(&client, |dual_x, c| dual_mul_scalar(dual_x, 2.0, c), &x)
                .unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Should be diagonal with 2s
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 2.0).abs() < 1e-10);
        assert!((j[8] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_quadratic() {
        let (device, client) = setup();

        // F(x) = x² (element-wise)
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let jacobian =
            jacobian_forward_impl(&client, |dual_x, c| dual_mul(dual_x, dual_x, c), &x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Should be diagonal with [2, 4, 6]
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 4.0).abs() < 1e-10);
        assert!((j[8] - 6.0).abs() < 1e-10);
    }
}
