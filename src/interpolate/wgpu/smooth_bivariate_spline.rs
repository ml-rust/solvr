use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::smooth_bivariate_spline::{
    smooth_bivariate_spline_evaluate_impl, smooth_bivariate_spline_fit_impl,
};
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use crate::interpolate::traits::smooth_bivariate_spline::SmoothBivariateSplineAlgorithms;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + UtilityOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
> SmoothBivariateSplineAlgorithms<R> for C
{
    fn smooth_bivariate_spline_fit(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        z: &Tensor<R>,
        weights: Option<&Tensor<R>>,
        smoothing: f64,
        kx: usize,
        ky: usize,
    ) -> InterpolateResult<BivariateSpline<R>> {
        smooth_bivariate_spline_fit_impl(self, x, y, z, weights, smoothing, kx, ky)
    }

    fn smooth_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<R>,
        xi: &Tensor<R>,
        yi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        smooth_bivariate_spline_evaluate_impl(self, spline, xi, yi)
    }
}
