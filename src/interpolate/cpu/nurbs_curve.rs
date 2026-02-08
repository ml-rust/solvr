use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_curve::{
    nurbs_curve_derivative_impl, nurbs_curve_evaluate_impl, nurbs_curve_subdivide_impl,
};
use crate::interpolate::traits::nurbs_curve::{NurbsCurve, NurbsCurveAlgorithms};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl NurbsCurveAlgorithms<CpuRuntime> for CpuClient {
    fn nurbs_curve_evaluate(
        &self,
        curve: &NurbsCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        nurbs_curve_evaluate_impl(self, curve, t)
    }

    fn nurbs_curve_derivative(
        &self,
        curve: &NurbsCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        nurbs_curve_derivative_impl(self, curve, t, order)
    }

    fn nurbs_curve_subdivide(
        &self,
        curve: &NurbsCurve<CpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(NurbsCurve<CpuRuntime>, NurbsCurve<CpuRuntime>)> {
        nurbs_curve_subdivide_impl(self, curve, t)
    }
}
