use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline_surface::{
    bspline_surface_evaluate_impl, bspline_surface_normal_impl, bspline_surface_partial_impl,
};
use crate::interpolate::traits::bspline_surface::{BSplineSurface, BSplineSurfaceAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BSplineSurfaceAlgorithms<WgpuRuntime> for WgpuClient {
    fn bspline_surface_evaluate(
        &self,
        surface: &BSplineSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_surface_evaluate_impl(self, surface, u, v)
    }

    fn bspline_surface_partial(
        &self,
        surface: &BSplineSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn bspline_surface_normal(
        &self,
        surface: &BSplineSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bspline_surface_normal_impl(self, surface, u, v)
    }
}
