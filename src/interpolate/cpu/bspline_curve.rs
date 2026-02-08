use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_curve::{
    bspline_curve_derivative_impl, bspline_curve_evaluate_impl, bspline_curve_subdivide_impl,
};
use crate::interpolate::traits::bspline_curve::{BSplineCurve, BSplineCurveAlgorithms};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BSplineCurveAlgorithms<CpuRuntime> for CpuClient {
    fn bspline_curve_evaluate(
        &self,
        curve: &BSplineCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bspline_curve_evaluate_impl(self, curve, t)
    }

    fn bspline_curve_derivative(
        &self,
        curve: &BSplineCurve<CpuRuntime>,
        t: &Tensor<CpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        bspline_curve_derivative_impl(self, curve, t, order)
    }

    fn bspline_curve_subdivide(
        &self,
        curve: &BSplineCurve<CpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(BSplineCurve<CpuRuntime>, BSplineCurve<CpuRuntime>)> {
        bspline_curve_subdivide_impl(self, curve, t)
    }
}
