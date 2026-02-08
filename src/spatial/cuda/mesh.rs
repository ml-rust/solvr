//! CUDA implementation of mesh algorithms.

use crate::spatial::impl_generic::mesh::{
    mesh_simplify_impl, mesh_smooth_impl, triangulate_polygon_impl,
};
use crate::spatial::traits::mesh::{Mesh, MeshAlgorithms, SimplificationMethod, SmoothingMethod};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MeshAlgorithms<CudaRuntime> for CudaClient {
    fn triangulate_polygon(&self, vertices: &Tensor<CudaRuntime>) -> Result<Mesh<CudaRuntime>> {
        triangulate_polygon_impl(self, vertices)
    }

    fn mesh_simplify(
        &self,
        mesh: &Mesh<CudaRuntime>,
        target_faces: usize,
        method: SimplificationMethod,
    ) -> Result<Mesh<CudaRuntime>> {
        mesh_simplify_impl(self, mesh, target_faces, method)
    }

    fn mesh_smooth(
        &self,
        mesh: &Mesh<CudaRuntime>,
        iterations: usize,
        method: SmoothingMethod,
    ) -> Result<Mesh<CudaRuntime>> {
        mesh_smooth_impl(self, mesh, iterations, method)
    }
}
