//! WebGPU implementation of graph matrix algorithms.

use crate::graph::impl_generic::{
    adjacency_matrix_impl, incidence_matrix_impl, laplacian_matrix_impl,
};
use crate::graph::traits::matrices::GraphMatrixAlgorithms;
use crate::graph::traits::types::GraphData;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::sparse::SparseTensor;

impl GraphMatrixAlgorithms<WgpuRuntime> for WgpuClient {
    fn laplacian_matrix(
        &self,
        graph: &GraphData<WgpuRuntime>,
        normalized: bool,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        laplacian_matrix_impl(self, graph, normalized)
    }

    fn adjacency_matrix(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        adjacency_matrix_impl(self, graph)
    }

    fn incidence_matrix(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        incidence_matrix_impl(self, graph)
    }
}
