use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::cubic_spline::cubic_spline_coefficients;
use crate::interpolate::traits::cubic_spline::{CubicSplineAlgorithms, SplineBoundary};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CubicSplineAlgorithms<CudaRuntime> for CudaClient {
    fn cubic_spline_coefficients(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        boundary: &SplineBoundary,
    ) -> InterpolateResult<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        cubic_spline_coefficients(self, x, y, boundary)
    }
}
