//! 1D interpolation for WebGPU runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interp1d::interp1d_evaluate;
use crate::interpolate::traits::interp1d::{Interp1dAlgorithms, InterpMethod};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl Interp1dAlgorithms<WgpuRuntime> for WgpuClient {
    fn interp1d(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        x_new: &Tensor<WgpuRuntime>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        interp1d_evaluate(self, x, y, x_new, method)
    }
}
