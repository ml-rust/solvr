use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::cubic_spline::cubic_spline_coefficients;
use crate::interpolate::traits::cubic_spline::{CubicSplineAlgorithms, SplineBoundary};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CubicSplineAlgorithms<WgpuRuntime> for WgpuClient {
    fn cubic_spline_coefficients(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        boundary: &SplineBoundary,
    ) -> InterpolateResult<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        cubic_spline_coefficients(self, x, y, boundary)
    }
}
