//! Akima interpolation for CUDA runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::akima::akima_slopes;
use crate::interpolate::traits::akima::AkimaAlgorithms;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl AkimaAlgorithms<CudaRuntime> for CudaClient {
    fn akima_slopes(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        akima_slopes(self, x, y)
    }
}
