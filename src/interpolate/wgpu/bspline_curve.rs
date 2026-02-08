use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_curve::{
    bspline_curve_derivative_impl, bspline_curve_evaluate_impl, bspline_curve_subdivide_impl,
};
use crate::interpolate::traits::bspline_curve::{BSplineCurve, BSplineCurveAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BSplineCurveAlgorithms<WgpuRuntime> for WgpuClient {
    fn bspline_curve_evaluate(
        &self,
        curve: &BSplineCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_curve_evaluate_impl(self, curve, t)
    }

    fn bspline_curve_derivative(
        &self,
        curve: &BSplineCurve<WgpuRuntime>,
        t: &Tensor<WgpuRuntime>,
        order: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_curve_derivative_impl(self, curve, t, order)
    }

    fn bspline_curve_subdivide(
        &self,
        curve: &BSplineCurve<WgpuRuntime>,
        t: f64,
    ) -> InterpolateResult<(BSplineCurve<WgpuRuntime>, BSplineCurve<WgpuRuntime>)> {
        bspline_curve_subdivide_impl(self, curve, t)
    }
}
