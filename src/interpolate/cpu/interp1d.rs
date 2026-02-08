use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interp1d::interp1d_evaluate;
use crate::interpolate::traits::interp1d::{Interp1dAlgorithms, InterpMethod};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl Interp1dAlgorithms<CpuRuntime> for CpuClient {
    fn interp1d(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        x_new: &Tensor<CpuRuntime>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        interp1d_evaluate(self, x, y, x_new, method)
    }
}
