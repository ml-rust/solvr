use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_curve::{
    bezier_derivative_impl, bezier_evaluate_impl, bezier_subdivide_impl,
};
use crate::interpolate::traits::bezier_curve::{BezierCurve, BezierCurveAlgorithms};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BezierCurveAlgorithms<CpuRuntime> for CpuClient {
    fn bezier_evaluate(
        &self,
        curve: &BezierCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bezier_evaluate_impl(self, curve, t)
    }

    fn bezier_derivative(
        &self,
        curve: &BezierCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bezier_derivative_impl(self, curve, t, order)
    }

    fn bezier_subdivide(
        &self,
        curve: &BezierCurve<CpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(BezierCurve<CpuRuntime>, BezierCurve<CpuRuntime>)> {
        bezier_subdivide_impl(self, curve, t)
    }
}
