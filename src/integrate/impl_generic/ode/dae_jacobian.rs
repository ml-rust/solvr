//! Jacobian computation for DAE solvers.
//!
//! For a DAE F(t, y, y') = 0, the Newton iteration requires the combined Jacobian:
//!
//!   J = ∂F/∂y + α * ∂F/∂y'
//!
//! where α = α₀/(h·β) comes from the BDF formula y'_{n+1} = (α₀·y_{n+1} + ...)/( h·β).
//!
//! This module provides autograd-based computation of both partial Jacobians.
use crate::DType;

use numr::autograd::DualTensor;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::common::jacobian::jacobian_autograd;

/// Compute the combined DAE Jacobian: J = ∂F/∂y + coeff * ∂F/∂y'.
///
/// For the BDF method applied to F(t, y, y') = 0, we substitute:
///   y'_{n+1} = (α₀·y_{n+1} + Σⱼ αⱼ·y_{n+1-j}) / (h·β)
///
/// The Newton iteration solves G(y) = F(t, y, y'(y)) = 0, requiring:
///   ∂G/∂y = ∂F/∂y + (∂F/∂y') * (dy'/dy) = ∂F/∂y + (α₀/(h·β)) * ∂F/∂y'
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - DAE residual function F(t, y, y') using DualTensor ops
/// * `t` - Current time (regular Tensor)
/// * `y` - Current state
/// * `yp` - Current derivative estimate
/// * `coeff` - Coefficient α₀/(h·β) from BDF formula
///
/// # Returns
///
/// Combined Jacobian matrix [n, n]
pub fn compute_dae_jacobian<R, C, F>(
    client: &C,
    f: &F,
    t: &Tensor<R>,
    y: &Tensor<R>,
    yp: &Tensor<R>,
    coeff: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    // Wrap t in DualTensor with no tangent (not differentiating w.r.t. time)
    let t_dual = DualTensor::new(t.clone(), None);
    let yp_const = DualTensor::new(yp.clone(), None);
    let y_const = DualTensor::new(y.clone(), None);

    // J_y = ∂F/∂y (yp held constant)
    let j_y = jacobian_autograd(client, |y_dual, c| f(&t_dual, y_dual, &yp_const, c), y)?;

    // J_yp = ∂F/∂y' (y held constant)
    // Need fresh t_dual since the closure moved the previous one
    let t_dual_2 = DualTensor::new(t.clone(), None);
    let j_yp = jacobian_autograd(client, |yp_dual, c| f(&t_dual_2, &y_const, yp_dual, c), yp)?;

    // Combined: J = J_y + coeff * J_yp
    let scaled_j_yp = client.mul_scalar(&j_yp, coeff)?;
    client.add(&j_y, &scaled_j_yp)
}

/// Evaluate the DAE residual F(t, y, y') at primal values (no differentiation).
///
/// Wraps inputs in DualTensors with no tangent and extracts the primal result.
pub fn eval_dae_primal<R, C, F>(
    client: &C,
    f: &F,
    t: &Tensor<R>,
    y: &Tensor<R>,
    yp: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let t_dual = DualTensor::new(t.clone(), None);
    let y_dual = DualTensor::new(y.clone(), None);
    let yp_dual = DualTensor::new(yp.clone(), None);

    let result = f(&t_dual, &y_dual, &yp_dual, client)?;
    Ok(result.primal().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul_scalar, dual_sub};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_dae_jacobian_simple() {
        let (device, client) = setup();

        // DAE: F(t, y, y') = y' - 2*y = 0  (exponential decay dy/dt = 2y)
        // ∂F/∂y = -2*I (derivative of y' - 2*y w.r.t. y is -2)
        // ∂F/∂y' = I (derivative of y' - 2*y w.r.t. y' is 1)
        // Combined with coeff=1.0: J = -2*I + 1.0*I = -I

        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device).unwrap();
        let yp = Tensor::<CpuRuntime>::from_slice(&[2.0, 4.0], &[2], &device).unwrap();

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| {
            // F = y' - 2*y
            let two_y = dual_mul_scalar(y, 2.0, c)?;
            dual_sub(yp, &two_y, c)
        };

        let jac = compute_dae_jacobian(&client, &f, &t, &y, &yp, 1.0).unwrap();
        let jac_data: Vec<f64> = jac.to_vec();

        // J_y = ∂(y' - 2*y)/∂y = -2*I
        // J_yp = ∂(y' - 2*y)/∂y' = I
        // J = J_y + coeff * J_yp = -2*I + 1*I = -I = [[-1, 0], [0, -1]]
        assert!(
            (jac_data[0] - (-1.0)).abs() < 1e-10,
            "J[0,0] = {}",
            jac_data[0]
        );
        assert!(jac_data[1].abs() < 1e-10, "J[0,1] = {}", jac_data[1]);
        assert!(jac_data[2].abs() < 1e-10, "J[1,0] = {}", jac_data[2]);
        assert!(
            (jac_data[3] - (-1.0)).abs() < 1e-10,
            "J[1,1] = {}",
            jac_data[3]
        );
    }

    #[test]
    fn test_eval_dae_primal() {
        let (device, client) = setup();

        // F(t, y, y') = y' - 2*y
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0], &[2], &device).unwrap();
        let yp = Tensor::<CpuRuntime>::from_slice(&[2.0, 6.0], &[2], &device).unwrap();

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| {
            let two_y = dual_mul_scalar(y, 2.0, c)?;
            dual_sub(yp, &two_y, c)
        };

        let residual = eval_dae_primal(&client, &f, &t, &y, &yp).unwrap();
        let res_data: Vec<f64> = residual.to_vec();

        // F = [2 - 2*1, 6 - 2*3] = [0, 0]
        assert!(res_data[0].abs() < 1e-10);
        assert!(res_data[1].abs() < 1e-10);
    }

    #[test]
    fn test_dae_jacobian_with_coeff() {
        let (device, client) = setup();

        // F = y' - y (simple: ∂F/∂y = -I, ∂F/∂y' = I)
        // With coeff=2.0: J = -I + 2*I = I

        let t = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let yp = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| { dual_sub(yp, y, c) };

        let jac = compute_dae_jacobian(&client, &f, &t, &y, &yp, 2.0).unwrap();
        let jac_data: Vec<f64> = jac.to_vec();

        // J = -1 + 2*1 = 1
        assert!((jac_data[0] - 1.0).abs() < 1e-10);
    }
}
