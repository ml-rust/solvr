//! CPU implementation of centrality algorithms.

use crate::graph::impl_generic::{
    betweenness_centrality_impl, closeness_centrality_impl, degree_centrality_impl,
    eigenvector_centrality_impl, pagerank_impl,
};
use crate::graph::traits::centrality::CentralityAlgorithms;
use crate::graph::traits::types::{EigCentralityOptions, GraphData, PageRankOptions};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CentralityAlgorithms<CpuRuntime> for CpuClient {
    fn degree_centrality(&self, graph: &GraphData<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        degree_centrality_impl(self, graph)
    }

    fn betweenness_centrality(
        &self,
        graph: &GraphData<CpuRuntime>,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        betweenness_centrality_impl(self, graph, normalized)
    }

    fn closeness_centrality(&self, graph: &GraphData<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        closeness_centrality_impl(self, graph)
    }

    fn eigenvector_centrality(
        &self,
        graph: &GraphData<CpuRuntime>,
        options: &EigCentralityOptions,
    ) -> Result<Tensor<CpuRuntime>> {
        eigenvector_centrality_impl(self, graph, options)
    }

    fn pagerank(
        &self,
        graph: &GraphData<CpuRuntime>,
        options: &PageRankOptions,
    ) -> Result<Tensor<CpuRuntime>> {
        pagerank_impl(self, graph, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_degree_centrality() {
        let (client, device) = setup();

        // Complete graph K3
        let graph = GraphData::from_edge_list::<f64>(
            &[0, 0, 1, 1, 2, 2],
            &[1, 2, 0, 2, 0, 1],
            None,
            3,
            true,
            &device,
        )
        .unwrap();

        let centrality = client.degree_centrality(&graph).unwrap();
        let vals: Vec<f64> = centrality.to_vec();
        // Each node connected to 2 others, centrality = 2/(3-1) = 1.0
        for v in &vals {
            assert!((*v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pagerank() {
        let (client, device) = setup();

        // Simple chain: 0 -> 1 -> 2
        let graph =
            GraphData::from_edge_list::<f64>(&[0, 1], &[1, 2], None, 3, true, &device).unwrap();

        let options = PageRankOptions::default();
        let pr = client.pagerank(&graph, &options).unwrap();
        let vals: Vec<f64> = pr.to_vec();

        // Sum should be ~1.0
        let sum: f64 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Node 2 should have highest rank (sink node absorbs)
        assert!(vals[2] > vals[1]);
    }

    #[test]
    fn test_eigenvector_centrality() {
        let (client, device) = setup();

        // Star graph: 0 connected to 1,2,3
        let graph =
            GraphData::from_edge_list::<f64>(&[0, 0, 0], &[1, 2, 3], None, 4, false, &device)
                .unwrap();

        let options = EigCentralityOptions::default();
        let centrality = client.eigenvector_centrality(&graph, &options).unwrap();
        let vals: Vec<f64> = centrality.to_vec();

        // Center node should have highest centrality
        // (or at least as high as leaf nodes)
        assert!(vals[0] >= vals[1]);
        assert!(vals[0] >= vals[2]);
        assert!(vals[0] >= vals[3]);
        // All values should be positive
        for v in &vals {
            assert!(*v >= 0.0);
        }
    }
}
