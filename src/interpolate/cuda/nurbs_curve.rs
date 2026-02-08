use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_curve::{
    nurbs_curve_derivative_impl, nurbs_curve_evaluate_impl, nurbs_curve_subdivide_impl,
};
use crate::interpolate::traits::nurbs_curve::{NurbsCurve, NurbsCurveAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl NurbsCurveAlgorithms<CudaRuntime> for CudaClient {
    fn nurbs_curve_evaluate(
        &self,
        curve: &NurbsCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        nurbs_curve_evaluate_impl(self, curve, t)
    }

    fn nurbs_curve_derivative(
        &self,
        curve: &NurbsCurve<CudaRuntime>,
        t: &Tensor<CudaRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        nurbs_curve_derivative_impl(self, curve, t, order)
    }

    fn nurbs_curve_subdivide(
        &self,
        curve: &NurbsCurve<CudaRuntime>,
        t: f64,
    ) -> InterpolateResult<(NurbsCurve<CudaRuntime>, NurbsCurve<CudaRuntime>)> {
        nurbs_curve_subdivide_impl(self, curve, t)
    }
}
