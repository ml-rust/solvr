use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::smooth_bivariate_spline::{
    smooth_bivariate_spline_evaluate_impl, smooth_bivariate_spline_fit_impl,
};
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use crate::interpolate::traits::smooth_bivariate_spline::SmoothBivariateSplineAlgorithms;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SmoothBivariateSplineAlgorithms<CudaRuntime> for CudaClient {
    fn smooth_bivariate_spline_fit(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        z: &Tensor<CudaRuntime>,
        weights: Option<&Tensor<CudaRuntime>>,
        smoothing: f64,
        kx: usize,
        ky: usize,
    ) -> InterpolateResult<BivariateSpline<CudaRuntime>> {
        smooth_bivariate_spline_fit_impl(self, x, y, z, weights, smoothing, kx, ky)
    }

    fn smooth_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<CudaRuntime>,
        xi: &Tensor<CudaRuntime>,
        yi: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        smooth_bivariate_spline_evaluate_impl(self, spline, xi, yi)
    }
}
