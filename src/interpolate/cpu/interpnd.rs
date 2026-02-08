use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interpnd::interpnd_evaluate;
use crate::interpolate::traits::interpnd::{ExtrapolateMode, InterpNdAlgorithms, InterpNdMethod};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl InterpNdAlgorithms<CpuRuntime> for CpuClient {
    fn interpnd(
        &self,
        points: &[&Tensor<CpuRuntime>],
        values: &Tensor<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        interpnd_evaluate(self, points, values, xi, method, extrapolate)
    }
}
