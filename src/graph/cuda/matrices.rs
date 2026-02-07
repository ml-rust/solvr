//! CUDA implementation of graph matrix algorithms.

use crate::graph::impl_generic::{
    adjacency_matrix_impl, incidence_matrix_impl, laplacian_matrix_impl,
};
use crate::graph::traits::matrices::GraphMatrixAlgorithms;
use crate::graph::traits::types::GraphData;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::sparse::SparseTensor;

impl GraphMatrixAlgorithms<CudaRuntime> for CudaClient {
    fn laplacian_matrix(
        &self,
        graph: &GraphData<CudaRuntime>,
        normalized: bool,
    ) -> Result<SparseTensor<CudaRuntime>> {
        laplacian_matrix_impl(self, graph, normalized)
    }

    fn adjacency_matrix(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<SparseTensor<CudaRuntime>> {
        adjacency_matrix_impl(self, graph)
    }

    fn incidence_matrix(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<SparseTensor<CudaRuntime>> {
        incidence_matrix_impl(self, graph)
    }
}
