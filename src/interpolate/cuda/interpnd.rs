//! N-dimensional grid interpolation for CUDA runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interpnd::interpnd_evaluate;
use crate::interpolate::traits::interpnd::{ExtrapolateMode, InterpNdAlgorithms, InterpNdMethod};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl InterpNdAlgorithms<CudaRuntime> for CudaClient {
    fn interpnd(
        &self,
        points: &[&Tensor<CudaRuntime>],
        values: &Tensor<CudaRuntime>,
        xi: &Tensor<CudaRuntime>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        interpnd_evaluate(self, points, values, xi, method, extrapolate)
    }
}
