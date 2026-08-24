//! CPU implementation of SOCP.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::socp_impl;
use crate::optimize::conic::traits::socp::{
    SocConstraint, SocpAlgorithms, SocpOptions, SocpResult,
};
use crate::optimize::error::OptimizeResult;

impl SocpAlgorithms<CpuRuntime> for CpuClient {
    fn solve_socp(
        &self,
        c: &Tensor<CpuRuntime>,
        constraints: &[SocConstraint<CpuRuntime>],
        options: &SocpOptions,
    ) -> OptimizeResult<SocpResult<CpuRuntime>> {
        socp_impl(self, c, constraints, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_socp_unconstrained() {
        let (_device, client) = setup();
        let device = _device.clone();
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device).unwrap();

        let result = client.solve_socp(&c, &[], &SocpOptions::default()).unwrap();

        assert!(result.converged);
        assert!(result.fun.abs() < 1e-6);
    }

    #[test]
    fn test_socp_simple_constraint() {
        let (_device, client) = setup();
        let device = _device.clone();
        // min c'*x = [1, 1]'*x s.t. ||[1, 1]'*x + 0|| <= 1'*x + 0
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let c_cone = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let d = 0.0;

        let constraint = SocConstraint { a, b, c: c_cone, d };

        let result = client
            .solve_socp(&c, &[constraint], &SocpOptions::default())
            .unwrap();

        assert!(result.iterations > 0);
        assert!(result.fun < 10.0); // Objective should be bounded
    }
}
