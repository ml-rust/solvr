use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::rect_bivariate_spline::{
    rect_bivariate_spline_evaluate_grid_impl, rect_bivariate_spline_evaluate_impl,
    rect_bivariate_spline_fit_impl, rect_bivariate_spline_integrate_impl,
    rect_bivariate_spline_partial_derivative_impl,
};
use crate::interpolate::traits::bspline::BSplineBoundary;
use crate::interpolate::traits::rect_bivariate_spline::{
    BivariateSpline, RectBivariateSplineAlgorithms,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RectBivariateSplineAlgorithms<CpuRuntime> for CpuClient {
    fn rect_bivariate_spline_fit(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        z: &Tensor<CpuRuntime>,
        degree_x: usize,
        degree_y: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BivariateSpline<CpuRuntime>> {
        rect_bivariate_spline_fit_impl(self, x, y, z, degree_x, degree_y, boundary)
    }

    fn rect_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        yi: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rect_bivariate_spline_evaluate_impl(self, spline, xi, yi)
    }

    fn rect_bivariate_spline_evaluate_grid(
        &self,
        spline: &BivariateSpline<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        yi: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rect_bivariate_spline_evaluate_grid_impl(self, spline, xi, yi)
    }

    fn rect_bivariate_spline_partial_derivative(
        &self,
        spline: &BivariateSpline<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        yi: &Tensor<CpuRuntime>,
        dx: usize,
        dy: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rect_bivariate_spline_partial_derivative_impl(self, spline, xi, yi, dx, dy)
    }

    fn rect_bivariate_spline_integrate(
        &self,
        spline: &BivariateSpline<CpuRuntime>,
        xa: f64,
        xb: f64,
        ya: f64,
        yb: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rect_bivariate_spline_integrate_impl(self, spline, xa, xb, ya, yb)
    }
}
