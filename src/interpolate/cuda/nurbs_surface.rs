use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::nurbs_surface::{
    nurbs_surface_evaluate_impl, nurbs_surface_normal_impl, nurbs_surface_partial_impl,
};
use crate::interpolate::traits::nurbs_surface::{NurbsSurface, NurbsSurfaceAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl NurbsSurfaceAlgorithms<CudaRuntime> for CudaClient {
    fn nurbs_surface_evaluate(
        &self,
        surface: &NurbsSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        nurbs_surface_evaluate_impl(self, surface, u, v)
    }

    fn nurbs_surface_partial(
        &self,
        surface: &NurbsSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        nurbs_surface_partial_impl(self, surface, u, v, du, dv)
    }

    fn nurbs_surface_normal(
        &self,
        surface: &NurbsSurface<CudaRuntime>,
        u: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        nurbs_surface_normal_impl(self, surface, u, v)
    }
}
