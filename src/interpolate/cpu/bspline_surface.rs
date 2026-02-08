use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_surface::{
    bspline_surface_evaluate_impl, bspline_surface_normal_impl, bspline_surface_partial_impl,
};
use crate::interpolate::traits::bspline_surface::{BSplineSurface, BSplineSurfaceAlgorithms};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>>
    BSplineSurfaceAlgorithms<R> for C
{
    fn bspline_surface_evaluate(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_surface_evaluate_impl(self, surface, u, v)
    }

    fn bspline_surface_partial(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn bspline_surface_normal(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_surface_normal_impl(self, surface, u, v)
    }
}
