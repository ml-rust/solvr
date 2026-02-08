use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_surface::{
    bspline_surface_evaluate_impl, bspline_surface_normal_impl, bspline_surface_partial_impl,
};
use crate::interpolate::traits::bspline_surface::{BSplineSurface, BSplineSurfaceAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BSplineSurfaceAlgorithms<CudaRuntime> for CudaClient {
    fn bspline_surface_evaluate(
        &self,
        surface: &BSplineSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_surface_evaluate_impl(self, surface, u, v)
    }

    fn bspline_surface_partial(
        &self,
        surface: &BSplineSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn bspline_surface_normal(
        &self,
        surface: &BSplineSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        bspline_surface_normal_impl(self, surface, u, v)
    }
}
