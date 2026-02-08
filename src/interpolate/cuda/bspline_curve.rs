use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_curve::{
    bspline_curve_derivative_impl, bspline_curve_evaluate_impl, bspline_curve_subdivide_impl,
};
use crate::interpolate::traits::bspline_curve::{BSplineCurve, BSplineCurveAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BSplineCurveAlgorithms<CudaRuntime> for CudaClient {
    fn bspline_curve_evaluate(
        &self,
        curve: &BSplineCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_curve_evaluate_impl(self, curve, t)
    }

    fn bspline_curve_derivative(
        &self,
        curve: &BSplineCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_curve_derivative_impl(self, curve, t, order)
    }

    fn bspline_curve_subdivide(
        &self,
        curve: &BSplineCurve<CudaRuntime>,
        t: f64,
    ) -> InterpolateResult<(BSplineCurve<CudaRuntime>, BSplineCurve<CudaRuntime>)> {
        bspline_curve_subdivide_impl(self, curve, t)
    }
}
