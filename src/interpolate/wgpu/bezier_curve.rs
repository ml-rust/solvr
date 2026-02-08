use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_curve::{
    bezier_derivative_impl, bezier_evaluate_impl, bezier_subdivide_impl,
};
use crate::interpolate::traits::bezier_curve::{BezierCurve, BezierCurveAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BezierCurveAlgorithms<WgpuRuntime> for WgpuClient {
    fn bezier_evaluate(
        &self,
        curve: &BezierCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bezier_evaluate_impl(self, curve, t)
    }

    fn bezier_derivative(
        &self,
        curve: &BezierCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bezier_derivative_impl(self, curve, t, order)
    }

    fn bezier_subdivide(
        &self,
        curve: &BezierCurve<WgpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(BezierCurve<WgpuRuntime>, BezierCurve<WgpuRuntime>)> {
        bezier_subdivide_impl(self, curve, t)
    }
}
