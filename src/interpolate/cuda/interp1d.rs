//! 1D interpolation for CUDA runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interp1d::interp1d_evaluate;
use crate::interpolate::traits::interp1d::{Interp1dAlgorithms, InterpMethod};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl Interp1dAlgorithms<CudaRuntime> for CudaClient {
    fn interp1d(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        x_new: &Tensor<CudaRuntime>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        interp1d_evaluate(self, x, y, x_new, method)
    }
}
