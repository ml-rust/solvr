use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::scattered::griddata_impl;
use crate::interpolate::traits::scattered::{ScatteredInterpAlgorithms, ScatteredMethod};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ScatteredInterpAlgorithms<CpuRuntime> for CpuClient {
    fn griddata(
        &self,
        points: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        method: ScatteredMethod,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        griddata_impl(self, points, values, xi, method)
    }
}
