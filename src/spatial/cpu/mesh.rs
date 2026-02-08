//! CPU implementation of mesh algorithms.

use crate::spatial::impl_generic::mesh::{
    mesh_simplify_impl, mesh_smooth_impl, triangulate_polygon_impl,
};
use crate::spatial::traits::mesh::{Mesh, MeshAlgorithms, SimplificationMethod, SmoothingMethod};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MeshAlgorithms<CpuRuntime> for CpuClient {
    fn triangulate_polygon(&self, vertices: &Tensor<CpuRuntime>) -> Result<Mesh<CpuRuntime>> {
        triangulate_polygon_impl(self, vertices)
    }

    fn mesh_simplify(
        &self,
        mesh: &Mesh<CpuRuntime>,
        target_faces: usize,
        method: SimplificationMethod,
    ) -> Result<Mesh<CpuRuntime>> {
        mesh_simplify_impl(self, mesh, target_faces, method)
    }

    fn mesh_smooth(
        &self,
        mesh: &Mesh<CpuRuntime>,
        iterations: usize,
        method: SmoothingMethod,
    ) -> Result<Mesh<CpuRuntime>> {
        mesh_smooth_impl(self, mesh, iterations, method)
    }
}
