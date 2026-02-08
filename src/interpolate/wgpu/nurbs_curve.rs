use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_curve::{
    nurbs_curve_derivative_impl, nurbs_curve_evaluate_impl, nurbs_curve_subdivide_impl,
};
use crate::interpolate::traits::nurbs_curve::{NurbsCurve, NurbsCurveAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl NurbsCurveAlgorithms<WgpuRuntime> for WgpuClient {
    fn nurbs_curve_evaluate(
        &self,
        curve: &NurbsCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        nurbs_curve_evaluate_impl(self, curve, t)
    }

    fn nurbs_curve_derivative(
        &self,
        curve: &NurbsCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        nurbs_curve_derivative_impl(self, curve, t, order)
    }

    fn nurbs_curve_subdivide(
        &self,
        curve: &NurbsCurve<WgpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(NurbsCurve<WgpuRuntime>, NurbsCurve<WgpuRuntime>)> {
        nurbs_curve_subdivide_impl(self, curve, t)
    }
}
