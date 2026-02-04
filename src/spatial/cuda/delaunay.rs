//! CUDA implementation of Delaunay triangulation.

use crate::spatial::impl_generic::{
    delaunay_find_simplex_impl, delaunay_impl, delaunay_vertex_neighbors_impl,
};
use crate::spatial::traits::delaunay::{Delaunay, DelaunayAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl DelaunayAlgorithms<CudaRuntime> for CudaClient {
    fn delaunay(&self, points: &Tensor<CudaRuntime>) -> Result<Delaunay<CudaRuntime>> {
        delaunay_impl(self, points)
    }

    fn delaunay_find_simplex(
        &self,
        tri: &Delaunay<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        delaunay_find_simplex_impl(self, tri, query)
    }

    fn delaunay_vertex_neighbors(
        &self,
        tri: &Delaunay<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        delaunay_vertex_neighbors_impl(self, tri)
    }
}
