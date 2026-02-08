//! N-dimensional grid interpolation for WebGPU runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interpnd::interpnd_evaluate;
use crate::interpolate::traits::interpnd::{ExtrapolateMode, InterpNdAlgorithms, InterpNdMethod};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl InterpNdAlgorithms<WgpuRuntime> for WgpuClient {
    fn interpnd(
        &self,
        points: &[&Tensor<WgpuRuntime>],
        values: &Tensor<WgpuRuntime>,
        xi: &Tensor<WgpuRuntime>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        interpnd_evaluate(self, points, values, xi, method, extrapolate)
    }
}
