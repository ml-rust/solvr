use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_curve::{
    bezier_derivative_impl, bezier_evaluate_impl, bezier_subdivide_impl,
};
use crate::interpolate::traits::bezier_curve::{BezierCurve, BezierCurveAlgorithms};
use numr::algorithm::special::SpecialFunctions;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
> BezierCurveAlgorithms<R> for C
{
    fn bezier_evaluate(
        &self,
        curve: &BezierCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bezier_evaluate_impl(self, curve, t)
    }

    fn bezier_derivative(
        &self,
        curve: &BezierCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>> {
        bezier_derivative_impl(self, curve, t, order)
    }

    fn bezier_subdivide(
        &self,
        curve: &BezierCurve<R>,
        t: f64,
    ) -> InterpolateResult<(BezierCurve<R>, BezierCurve<R>)> {
        bezier_subdivide_impl(self, curve, t)
    }
}
