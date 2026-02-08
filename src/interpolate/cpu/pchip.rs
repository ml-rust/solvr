use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::pchip::pchip_slopes;
use crate::interpolate::traits::pchip::PchipAlgorithms;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl PchipAlgorithms<CpuRuntime> for CpuClient {
    fn pchip_slopes(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        pchip_slopes(self, x, y)
    }
}
