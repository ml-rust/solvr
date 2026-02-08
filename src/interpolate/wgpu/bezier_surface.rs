use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_surface::{
    bezier_surface_evaluate_impl, bezier_surface_normal_impl, bezier_surface_partial_impl,
};
use crate::interpolate::traits::bezier_surface::{BezierSurface, BezierSurfaceAlgorithms};
use numr::algorithm::special::SpecialFunctions;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
> BezierSurfaceAlgorithms<R> for C
{
    fn bezier_surface_evaluate(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bezier_surface_evaluate_impl(self, surface, u, v)
    }

    fn bezier_surface_partial(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>> {
        bezier_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn bezier_surface_normal(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bezier_surface_normal_impl(self, surface, u, v)
    }
}
