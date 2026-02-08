use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline::{
    bspline_derivative_impl, bspline_evaluate_impl, bspline_integrate_impl, make_interp_spline_impl,
};
use crate::interpolate::traits::bspline::{BSpline, BSplineAlgorithms, BSplineBoundary};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BSplineAlgorithms<CpuRuntime> for CpuClient {
    fn make_interp_spline(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        degree: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BSpline<CpuRuntime>> {
        make_interp_spline_impl(self, x, y, degree, boundary)
    }

    fn bspline_evaluate(
        &self,
        spline: &BSpline<CpuRuntime>,
        x_new: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bspline_evaluate_impl(self, spline, x_new)
    }

    fn bspline_derivative(
        &self,
        spline: &BSpline<CpuRuntime>,
        x_new: &Tensor<CpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bspline_derivative_impl(self, spline, x_new, order)
    }

    fn bspline_integrate(
        &self,
        spline: &BSpline<CpuRuntime>,
        a: f64,
        b: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bspline_integrate_impl(self, spline, a, b)
    }
}
