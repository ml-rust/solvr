use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::akima::akima_slopes;
use crate::interpolate::traits::akima::AkimaAlgorithms;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AkimaAlgorithms<CpuRuntime> for CpuClient {
    fn akima_slopes(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        akima_slopes(self, x, y)
    }
}
