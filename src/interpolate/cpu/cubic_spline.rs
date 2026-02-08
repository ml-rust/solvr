use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::cubic_spline::cubic_spline_coefficients;
use crate::interpolate::traits::cubic_spline::{CubicSplineAlgorithms, SplineBoundary};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CubicSplineAlgorithms<CpuRuntime> for CpuClient {
    fn cubic_spline_coefficients(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        boundary: &SplineBoundary,
    ) -> InterpolateResult<(
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
    )> {
        cubic_spline_coefficients(self, x, y, boundary)
    }
}
