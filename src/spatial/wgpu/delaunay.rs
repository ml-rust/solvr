//! WebGPU implementation of Delaunay triangulation.

use crate::spatial::impl_generic::{
    delaunay_find_simplex_impl, delaunay_impl, delaunay_vertex_neighbors_impl,
};
use crate::spatial::traits::delaunay::{Delaunay, DelaunayAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl DelaunayAlgorithms<WgpuRuntime> for WgpuClient {
    fn delaunay(&self, points: &Tensor<WgpuRuntime>) -> Result<Delaunay<WgpuRuntime>> {
        delaunay_impl(self, points)
    }

    fn delaunay_find_simplex(
        &self,
        tri: &Delaunay<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        delaunay_find_simplex_impl(self, tri, query)
    }

    fn delaunay_vertex_neighbors(
        &self,
        tri: &Delaunay<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        delaunay_vertex_neighbors_impl(self, tri)
    }
}
