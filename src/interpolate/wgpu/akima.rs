//! Akima interpolation for WebGPU runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::akima::akima_slopes;
use crate::interpolate::traits::akima::AkimaAlgorithms;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl AkimaAlgorithms<WgpuRuntime> for WgpuClient {
    fn akima_slopes(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        akima_slopes(self, x, y)
    }
}
