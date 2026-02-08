use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_curve::{
    bspline_curve_derivative_impl, bspline_curve_evaluate_impl, bspline_curve_subdivide_impl,
};
use crate::interpolate::traits::bspline_curve::{BSplineCurve, BSplineCurveAlgorithms};
use numr::ops::{
    CompareOps, ConditionalOps, ReduceOps, ScalarOps, SortingOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + SortingOps<R>
        + ReduceOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
> BSplineCurveAlgorithms<R> for C
{
    fn bspline_curve_evaluate(
        &self,
        curve: &BSplineCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_curve_evaluate_impl(self, curve, t)
    }

    fn bspline_curve_derivative(
        &self,
        curve: &BSplineCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_curve_derivative_impl(self, curve, t, order)
    }

    fn bspline_curve_subdivide(
        &self,
        curve: &BSplineCurve<R>,
        t: f64,
    ) -> InterpolateResult<(BSplineCurve<R>, BSplineCurve<R>)> {
        bspline_curve_subdivide_impl(self, curve, t)
    }
}
