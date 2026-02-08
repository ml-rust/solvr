use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::scattered::griddata_impl;
use crate::interpolate::traits::scattered::{ScatteredInterpAlgorithms, ScatteredMethod};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ScatteredInterpAlgorithms<CudaRuntime> for CudaClient {
    fn griddata(
        &self,
        points: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        xi: &Tensor<CudaRuntime>,
        method: ScatteredMethod,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        griddata_impl(self, points, values, xi, method)
    }
}
