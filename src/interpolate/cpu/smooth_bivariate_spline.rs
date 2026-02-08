use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::smooth_bivariate_spline::{
    smooth_bivariate_spline_evaluate_impl, smooth_bivariate_spline_fit_impl,
};
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use crate::interpolate::traits::smooth_bivariate_spline::SmoothBivariateSplineAlgorithms;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SmoothBivariateSplineAlgorithms<CpuRuntime> for CpuClient {
    fn smooth_bivariate_spline_fit(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        z: &Tensor<CpuRuntime>,
        weights: Option<&Tensor<CpuRuntime>>,
        smoothing: f64,
        kx: usize,
        ky: usize,
    ) -> InterpolateResult<BivariateSpline<CpuRuntime>> {
        smooth_bivariate_spline_fit_impl(self, x, y, z, weights, smoothing, kx, ky)
    }

    fn smooth_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
        yi: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        smooth_bivariate_spline_evaluate_impl(self, spline, xi, yi)
    }
}
