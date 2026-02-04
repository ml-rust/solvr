//! WebGPU implementation of Voronoi diagram.

use crate::spatial::impl_generic::{
    voronoi_find_region_impl, voronoi_from_delaunay_impl, voronoi_impl,
};
use crate::spatial::traits::delaunay::Delaunay;
use crate::spatial::traits::voronoi::{Voronoi, VoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl VoronoiAlgorithms<WgpuRuntime> for WgpuClient {
    fn voronoi(&self, points: &Tensor<WgpuRuntime>) -> Result<Voronoi<WgpuRuntime>> {
        voronoi_impl(self, points)
    }

    fn voronoi_from_delaunay(&self, tri: &Delaunay<WgpuRuntime>) -> Result<Voronoi<WgpuRuntime>> {
        voronoi_from_delaunay_impl(self, tri)
    }

    fn voronoi_find_region(
        &self,
        vor: &Voronoi<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        voronoi_find_region_impl(self, vor, query)
    }
}
