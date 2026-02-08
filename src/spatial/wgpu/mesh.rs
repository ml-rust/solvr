//! WebGPU implementation of mesh algorithms.

use crate::spatial::impl_generic::mesh::{
    mesh_simplify_impl, mesh_smooth_impl, triangulate_polygon_impl,
};
use crate::spatial::traits::mesh::{Mesh, MeshAlgorithms, SimplificationMethod, SmoothingMethod};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MeshAlgorithms<WgpuRuntime> for WgpuClient {
    fn triangulate_polygon(&self, vertices: &Tensor<WgpuRuntime>) -> Result<Mesh<WgpuRuntime>> {
        triangulate_polygon_impl(self, vertices)
    }

    fn mesh_simplify(
        &self,
        mesh: &Mesh<WgpuRuntime>,
        target_faces: usize,
        method: SimplificationMethod,
    ) -> Result<Mesh<WgpuRuntime>> {
        mesh_simplify_impl(self, mesh, target_faces, method)
    }

    fn mesh_smooth(
        &self,
        mesh: &Mesh<WgpuRuntime>,
        iterations: usize,
        method: SmoothingMethod,
    ) -> Result<Mesh<WgpuRuntime>> {
        mesh_smooth_impl(self, mesh, iterations, method)
    }
}
