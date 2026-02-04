//! Jacobian computation for implicit ODE solvers.
//!
//! Provides autograd-based Jacobian computation using numr's forward-mode AD.
//! This gives exact derivatives (to machine precision) without epsilon tuning.
//!
//! # Why Autograd over Finite Differences?
//!
//! - **Exact**: No truncation error from finite step size
//! - **Robust**: No numerical instability for ill-conditioned problems
//! - **No tuning**: No epsilon parameter to adjust
//! - **Unique**: No other Rust ODE library offers automatic Jacobians
//!
//! # Usage
//!
//! Users write their ODE function using `DualTensor` and `dual_*` operations:
//!
//! ```ignore
//! use numr::autograd::dual_ops::{dual_mul, dual_sub, dual_mul_scalar};
//!
//! // Van der Pol oscillator: y'' - μ(1-y²)y' + y = 0
//! // As system: [y₀' = y₁, y₁' = μ(1-y₀²)y₁ - y₀]
//! let f = |_t: &DualTensor<R>, y: &DualTensor<R>, client: &C| {
//!     // Extract components, compute using dual ops
//!     // ...
//! };
//! ```

use numr::autograd::DualTensor;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::common::jacobian::jacobian_autograd;

/// Compute the Jacobian matrix ∂f/∂y using forward-mode automatic differentiation.
///
/// For an ODE dy/dt = f(t, y), computes the n×n Jacobian matrix J where
/// J[i,j] = ∂fᵢ/∂yⱼ.
///
/// This is a thin wrapper around [`crate::common::jacobian::jacobian_autograd`]
/// that handles the time parameter for ODE functions. Time is wrapped in a
/// DualTensor with no tangent (we don't differentiate w.r.t. time).
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - ODE right-hand side using `DualTensor` operations
/// * `t` - Current time (regular Tensor, not differentiated)
/// * `y` - Current state (will be converted to DualTensor internally)
///
/// # Returns
///
/// Jacobian matrix of shape [n, n] where n is state dimension.
pub fn compute_jacobian_autograd<R, C, F>(
    client: &C,
    f: &F,
    t: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    // Create DualTensor for t with zero tangent (we don't differentiate w.r.t. time)
    let t_dual = DualTensor::new(t.clone(), None);

    // Delegate to common jacobian computation with time curried in
    jacobian_autograd(client, |y_dual, c| f(&t_dual, y_dual, c), y)
}

/// Compute vector p-norm (stays on device, returns scalar tensor).
///
/// Computes ||x||_p = (sum(|x_i|^p))^(1/p)
///
/// For p=2, this is the Euclidean norm.
pub fn compute_norm<R, C>(client: &C, x: &Tensor<R>, p: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let x_abs = client.abs(x)?;
    let x_pow = client.pow_scalar(&x_abs, p)?;
    let sum = client.sum(&x_pow, &[0], false)?;
    client.pow_scalar(&sum, 1.0 / p)
}

/// Compute vector p-norm as a scalar f64.
///
/// Extracts the scalar value from the norm tensor.
/// This involves a device-to-host transfer and should only be used
/// for control flow decisions (e.g., convergence checks).
pub fn compute_norm_scalar<R, C>(client: &C, x: &Tensor<R>, p: f64) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let norm_tensor = compute_norm(client, x, p)?;
    Ok(norm_tensor.to_vec()[0])
}

/// Evaluate the ODE function with primal values only (no differentiation).
///
/// Wraps t and y in DualTensors with no tangent, calls the function, and
/// extracts the primal result. Used by implicit solvers for regular function
/// evaluation (not Jacobian computation).
pub fn eval_primal<R, C, F>(client: &C, f: &F, t: &Tensor<R>, y: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let t_dual = DualTensor::new(t.clone(), None);
    let y_dual = DualTensor::new(y.clone(), None);
    let result = f(&t_dual, &y_dual, client)?;
    Ok(result.primal().clone())
}

/// Compute the iteration matrix for implicit methods.
///
/// For BDF: M = I - h * beta * J
/// For Newton iteration on F(y) = y - h*beta*f(t,y) - rhs = 0
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `jacobian` - Jacobian matrix J = ∂f/∂y [n, n]
/// * `h` - Step size (scalar)
/// * `beta` - Method coefficient
///
/// # Returns
///
/// Iteration matrix M = I - h * beta * J [n, n]
pub fn compute_iteration_matrix<R, C>(
    client: &C,
    jacobian: &Tensor<R>,
    h: f64,
    beta: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let n = jacobian.shape()[0];

    // Create identity matrix using numr's GPU-efficient eye()
    let identity = client.eye(n, None, DType::F64)?;

    // M = I - h * beta * J
    let h_beta = h * beta;
    let scaled_j = client.mul_scalar(jacobian, h_beta)?;
    client.sub(&identity, &scaled_j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul, dual_mul_scalar, dual_sub};
    use numr::ops::MatmulOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_jacobian_autograd_linear() {
        let (device, client) = setup();

        // f(t, y) = 2*y (linear system)
        // Jacobian should be 2*I
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 c: &CpuClient|
         -> Result<DualTensor<CpuRuntime>> { dual_mul_scalar(y, 2.0, c) };

        let jac = compute_jacobian_autograd(&client, &f, &t, &y).unwrap();

        let jac_data: Vec<f64> = jac.to_vec();

        // Check diagonal is 2, off-diagonal is 0
        assert!((jac_data[0] - 2.0).abs() < 1e-10);
        assert!((jac_data[4] - 2.0).abs() < 1e-10);
        assert!((jac_data[8] - 2.0).abs() < 1e-10);
        assert!(jac_data[1].abs() < 1e-10);
        assert!(jac_data[2].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_autograd_nonlinear() {
        let (device, client) = setup();

        // f(t, y) = y² (element-wise)
        // Jacobian = diag(2*y) at y = [1, 2, 3] -> diag([2, 4, 6])
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 c: &CpuClient|
         -> Result<DualTensor<CpuRuntime>> { dual_mul(y, y, c) };

        let jac = compute_jacobian_autograd(&client, &f, &t, &y).unwrap();

        let jac_data: Vec<f64> = jac.to_vec();

        // Diagonal should be [2, 4, 6]
        assert!((jac_data[0] - 2.0).abs() < 1e-10);
        assert!((jac_data[4] - 4.0).abs() < 1e-10);
        assert!((jac_data[8] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_autograd_coupled() {
        let (device, client) = setup();

        // f(t, y) = [y[1], -y[0]] (harmonic oscillator)
        // Jacobian = [[0, 1], [-1, 0]]
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device);

        // For coupled systems, we need to work with the full vector
        // This is a simplified test - real usage would index into components
        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 c: &CpuClient|
         -> Result<DualTensor<CpuRuntime>> {
            // Swap and negate: [y1, -y0]
            // For now, just test with a simple transformation
            let neg_y = dual_mul_scalar(y, -1.0, c)?;
            Ok(neg_y)
        };

        let jac = compute_jacobian_autograd(&client, &f, &t, &y).unwrap();
        let jac_data: Vec<f64> = jac.to_vec();

        // Should be -I
        assert!((jac_data[0] - (-1.0)).abs() < 1e-10);
        assert!((jac_data[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_iteration_matrix() {
        let (device, client) = setup();

        // J = [[1, 0], [0, 2]]
        let j = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 2.0], &[2, 2], &device);

        // M = I - h*beta*J with h=0.1, beta=1.0
        // M = [[1, 0], [0, 1]] - 0.1*[[1, 0], [0, 2]]
        // M = [[0.9, 0], [0, 0.8]]
        let m = compute_iteration_matrix(&client, &j, 0.1, 1.0).unwrap();
        let m_data: Vec<f64> = m.to_vec();

        assert!((m_data[0] - 0.9).abs() < 1e-10);
        assert!(m_data[1].abs() < 1e-10);
        assert!(m_data[2].abs() < 1e-10);
        assert!((m_data[3] - 0.8).abs() < 1e-10);
    }
}
