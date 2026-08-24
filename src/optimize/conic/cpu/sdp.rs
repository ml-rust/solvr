//! CPU implementation of SDP.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::sdp_impl;
use crate::optimize::conic::traits::sdp::{SdpAlgorithms, SdpOptions, SdpResult};
use crate::optimize::error::OptimizeResult;

impl SdpAlgorithms<CpuRuntime> for CpuClient {
    fn solve_sdp(
        &self,
        c_mat: &Tensor<CpuRuntime>,
        a_mats: &[Tensor<CpuRuntime>],
        b_vec: &Tensor<CpuRuntime>,
        options: &SdpOptions,
    ) -> OptimizeResult<SdpResult<CpuRuntime>> {
        sdp_impl(self, c_mat, a_mats, b_vec, options)
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
    fn test_sdp_basic() {
        let (_device, client) = setup();
        let device = _device.clone();
        // min trace(C*X) = trace([[1,0],[0,1]]*X) = x11 + x22
        // s.t. trace(A*X) = 1, where A = [[1, 0], [0, 1]]
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let result = client
            .solve_sdp(&c, &[a], &b, &SdpOptions::default())
            .unwrap();

        // Objective should be computed and iterations should be > 0
        assert!(result.iterations > 0);
    }
}
