//! PCHIP interpolation for CUDA runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::pchip::pchip_slopes;
use crate::interpolate::traits::pchip::PchipAlgorithms;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl PchipAlgorithms<CudaRuntime> for CudaClient {
    fn pchip_slopes(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        pchip_slopes(self, x, y)
    }
}
