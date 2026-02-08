use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::smooth_bivariate_spline::{
    smooth_bivariate_spline_evaluate_impl, smooth_bivariate_spline_fit_impl,
};
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use crate::interpolate::traits::smooth_bivariate_spline::SmoothBivariateSplineAlgorithms;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SmoothBivariateSplineAlgorithms<WgpuRuntime> for WgpuClient {
    fn smooth_bivariate_spline_fit(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        z: &Tensor<WgpuRuntime>,
        weights: Option<&Tensor<WgpuRuntime>>,
        smoothing: f64,
        kx: usize,
        ky: usize,
    ) -> InterpolateResult<BivariateSpline<WgpuRuntime>> {
        smooth_bivariate_spline_fit_impl(self, x, y, z, weights, smoothing, kx, ky)
    }

    fn smooth_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<WgpuRuntime>,
        xi: &Tensor<WgpuRuntime>,
        yi: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        smooth_bivariate_spline_evaluate_impl(self, spline, xi, yi)
    }
}
