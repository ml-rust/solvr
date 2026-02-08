//! PCHIP interpolation for WebGPU runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::pchip::pchip_slopes;
use crate::interpolate::traits::pchip::PchipAlgorithms;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl PchipAlgorithms<WgpuRuntime> for WgpuClient {
    fn pchip_slopes(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        pchip_slopes(self, x, y)
    }
}
