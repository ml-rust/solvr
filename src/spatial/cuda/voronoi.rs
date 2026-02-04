//! CUDA implementation of Voronoi diagram.

use crate::spatial::impl_generic::{
    voronoi_find_region_impl, voronoi_from_delaunay_impl, voronoi_impl,
};
use crate::spatial::traits::delaunay::Delaunay;
use crate::spatial::traits::voronoi::{Voronoi, VoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl VoronoiAlgorithms<CudaRuntime> for CudaClient {
    fn voronoi(&self, points: &Tensor<CudaRuntime>) -> Result<Voronoi<CudaRuntime>> {
        voronoi_impl(self, points)
    }

    fn voronoi_from_delaunay(&self, tri: &Delaunay<CudaRuntime>) -> Result<Voronoi<CudaRuntime>> {
        voronoi_from_delaunay_impl(self, tri)
    }

    fn voronoi_find_region(
        &self,
        vor: &Voronoi<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        voronoi_find_region_impl(self, vor, query)
    }
}
