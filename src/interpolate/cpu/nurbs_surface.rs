use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_surface::{
    nurbs_surface_evaluate_impl, nurbs_surface_normal_impl, nurbs_surface_partial_impl,
};
use crate::interpolate::traits::nurbs_surface::{NurbsSurface, NurbsSurfaceAlgorithms};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>>
    NurbsSurfaceAlgorithms<R> for C
{
    fn nurbs_surface_evaluate(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        nurbs_surface_evaluate_impl(self, surface, u, v)
    }

    fn nurbs_surface_partial(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>> {
        nurbs_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn nurbs_surface_normal(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        nurbs_surface_normal_impl(self, surface, u, v)
    }
}
