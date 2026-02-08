use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_curve::{
    nurbs_curve_derivative_impl, nurbs_curve_evaluate_impl, nurbs_curve_subdivide_impl,
};
use crate::interpolate::traits::nurbs_curve::{NurbsCurve, NurbsCurveAlgorithms};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>>
    NurbsCurveAlgorithms<R> for C
{
    fn nurbs_curve_evaluate(
        &self,
        curve: &NurbsCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        nurbs_curve_evaluate_impl(self, curve, t)
    }

    fn nurbs_curve_derivative(
        &self,
        curve: &NurbsCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>> {
        nurbs_curve_derivative_impl(self, curve, t, order)
    }

    fn nurbs_curve_subdivide(
        &self,
        curve: &NurbsCurve<R>,
        t: f64,
    ) -> InterpolateResult<(NurbsCurve<R>, NurbsCurve<R>)> {
        nurbs_curve_subdivide_impl(self, curve, t)
    }
}
