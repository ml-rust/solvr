use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_curve::{
    bezier_derivative_impl, bezier_evaluate_impl, bezier_subdivide_impl,
};
use crate::interpolate::traits::bezier_curve::{BezierCurve, BezierCurveAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BezierCurveAlgorithms<CudaRuntime> for CudaClient {
    fn bezier_evaluate(
        &self,
        curve: &BezierCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bezier_evaluate_impl(self, curve, t)
    }

    fn bezier_derivative(
        &self,
        curve: &BezierCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bezier_derivative_impl(self, curve, t, order)
    }

    fn bezier_subdivide(
        &self,
        curve: &BezierCurve<CudaRuntime>,
        t: f64,
    ) -> InterpolateResult<(BezierCurve<CudaRuntime>, BezierCurve<CudaRuntime>)> {
        bezier_subdivide_impl(self, curve, t)
    }
}
