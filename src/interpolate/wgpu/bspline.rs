use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline::{
    bspline_derivative_impl, bspline_evaluate_impl, bspline_integrate_impl, make_interp_spline_impl,
};
use crate::interpolate::traits::bspline::{BSpline, BSplineAlgorithms, BSplineBoundary};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BSplineAlgorithms<WgpuRuntime> for WgpuClient {
    fn make_interp_spline(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        degree: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BSpline<WgpuRuntime>> {
        make_interp_spline_impl(self, x, y, degree, boundary)
    }

    fn bspline_evaluate(
        &self,
        spline: &BSpline<WgpuRuntime>,
        x_new: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_evaluate_impl(self, spline, x_new)
    }

    fn bspline_derivative(
        &self,
        spline: &BSpline<WgpuRuntime>,
        x_new: &Tensor<WgpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_derivative_impl(self, spline, x_new, order)
    }

    fn bspline_integrate(
        &self,
        spline: &BSpline<WgpuRuntime>,
        a: f64,
        b: f64,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_integrate_impl(self, spline, a, b)
    }
}
