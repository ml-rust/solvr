use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline::{
    bspline_derivative_impl, bspline_evaluate_impl, bspline_integrate_impl, make_interp_spline_impl,
};
use crate::interpolate::traits::bspline::{BSpline, BSplineAlgorithms, BSplineBoundary};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BSplineAlgorithms<CudaRuntime> for CudaClient {
    fn make_interp_spline(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        degree: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BSpline<CudaRuntime>> {
        make_interp_spline_impl(self, x, y, degree, boundary)
    }

    fn bspline_evaluate(
        &self,
        spline: &BSpline<CudaRuntime>,
        x_new: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_evaluate_impl(self, spline, x_new)
    }

    fn bspline_derivative(
        &self,
        spline: &BSpline<CudaRuntime>,
        x_new: &Tensor<CudaRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_derivative_impl(self, spline, x_new, order)
    }

    fn bspline_integrate(
        &self,
        spline: &BSpline<CudaRuntime>,
        a: f64,
        b: f64,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_integrate_impl(self, spline, a, b)
    }
}
