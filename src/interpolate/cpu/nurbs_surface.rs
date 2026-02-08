use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_surface::{
    nurbs_surface_evaluate_impl, nurbs_surface_normal_impl, nurbs_surface_partial_impl,
};
use crate::interpolate::traits::nurbs_surface::{NurbsSurface, NurbsSurfaceAlgorithms};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl NurbsSurfaceAlgorithms<CpuRuntime> for CpuClient {
    fn nurbs_surface_evaluate(
        &self,
        surface: &NurbsSurface<CpuRuntime>,
        u: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        nurbs_surface_evaluate_impl(self, surface, u, v)
    }

    fn nurbs_surface_partial(
        &self,
        surface: &NurbsSurface<CpuRuntime>,
        u: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        nurbs_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn nurbs_surface_normal(
        &self,
        surface: &NurbsSurface<CpuRuntime>,
        u: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        nurbs_surface_normal_impl(self, surface, u, v)
    }
}
