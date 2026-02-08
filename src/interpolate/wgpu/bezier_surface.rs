use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bezier_surface::{
    bezier_surface_evaluate_impl, bezier_surface_normal_impl, bezier_surface_partial_impl,
};
use crate::interpolate::traits::bezier_surface::{BezierSurface, BezierSurfaceAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BezierSurfaceAlgorithms<WgpuRuntime> for WgpuClient {
    fn bezier_surface_evaluate(
        &self,
        surface: &BezierSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bezier_surface_evaluate_impl(self, surface, u, v)
    }

    fn bezier_surface_partial(
        &self,
        surface: &BezierSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bezier_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn bezier_surface_normal(
        &self,
        surface: &BezierSurface<WgpuRuntime>,
        u: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        bezier_surface_normal_impl(self, surface, u, v)
    }
}
