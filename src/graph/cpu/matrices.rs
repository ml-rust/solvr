//! CPU implementation of graph matrix algorithms.

use crate::graph::impl_generic::{
    adjacency_matrix_impl, incidence_matrix_impl, laplacian_matrix_impl,
};
use crate::graph::traits::matrices::GraphMatrixAlgorithms;
use crate::graph::traits::types::GraphData;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::sparse::SparseTensor;

impl GraphMatrixAlgorithms<CpuRuntime> for CpuClient {
    fn laplacian_matrix(
        &self,
        graph: &GraphData<CpuRuntime>,
        normalized: bool,
    ) -> Result<SparseTensor<CpuRuntime>> {
        laplacian_matrix_impl(self, graph, normalized)
    }

    fn adjacency_matrix(&self, graph: &GraphData<CpuRuntime>) -> Result<SparseTensor<CpuRuntime>> {
        adjacency_matrix_impl(self, graph)
    }

    fn incidence_matrix(&self, graph: &GraphData<CpuRuntime>) -> Result<SparseTensor<CpuRuntime>> {
        incidence_matrix_impl(self, graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_laplacian() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let graph =
            GraphData::from_edge_list::<f64>(&[0, 1], &[1, 2], None, 3, false, &device).unwrap();

        let lap = client.laplacian_matrix(&graph, false).unwrap();
        assert_eq!(lap.shape(), [3, 3]);
    }

    #[test]
    fn test_adjacency_matrix() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let graph =
            GraphData::from_edge_list::<f64>(&[0, 1], &[1, 2], None, 3, true, &device).unwrap();

        let adj = client.adjacency_matrix(&graph).unwrap();
        assert_eq!(adj.shape(), [3, 3]);
        assert_eq!(adj.nnz(), 2);
    }
}
