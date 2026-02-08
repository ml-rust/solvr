use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::scattered::griddata_impl;
use crate::interpolate::traits::scattered::{ScatteredInterpAlgorithms, ScatteredMethod};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ScatteredInterpAlgorithms<WgpuRuntime> for WgpuClient {
    fn griddata(
        &self,
        points: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        xi: &Tensor<WgpuRuntime>,
        method: ScatteredMethod,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        griddata_impl(self, points, values, xi, method)
    }
}
