//! CPU implementation of shortest path algorithms.

use crate::graph::impl_generic::{
    astar_impl, bellman_ford_impl, dijkstra_impl, floyd_warshall_impl, johnson_impl,
};
use crate::graph::traits::shortest_path::ShortestPathAlgorithms;
use crate::graph::traits::types::{AllPairsResult, GraphData, PathResult, ShortestPathResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ShortestPathAlgorithms<CpuRuntime> for CpuClient {
    fn dijkstra(
        &self,
        graph: &GraphData<CpuRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<CpuRuntime>> {
        dijkstra_impl(self, graph, source)
    }

    fn bellman_ford(
        &self,
        graph: &GraphData<CpuRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<CpuRuntime>> {
        bellman_ford_impl(self, graph, source)
    }

    fn floyd_warshall(&self, graph: &GraphData<CpuRuntime>) -> Result<AllPairsResult<CpuRuntime>> {
        floyd_warshall_impl(self, graph)
    }

    fn johnson(&self, graph: &GraphData<CpuRuntime>) -> Result<AllPairsResult<CpuRuntime>> {
        johnson_impl(self, graph)
    }

    fn astar(
        &self,
        graph: &GraphData<CpuRuntime>,
        source: usize,
        target: usize,
        heuristic: &Tensor<CpuRuntime>,
    ) -> Result<PathResult<CpuRuntime>> {
        astar_impl(self, graph, source, target, heuristic)
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

    fn make_test_graph(device: &CpuDevice) -> GraphData<CpuRuntime> {
        // Simple directed graph:
        // 0 --1--> 1 --2--> 2
        // |                  ^
        // +-------10---------+
        GraphData::from_edge_list::<f64>(
            &[0, 1, 0],
            &[1, 2, 2],
            Some(&[1.0, 2.0, 10.0]),
            3,
            true,
            device,
        )
        .unwrap()
    }

    #[test]
    fn test_dijkstra() {
        let (client, device) = setup();
        let graph = make_test_graph(&device);

        let result = client.dijkstra(&graph, 0).unwrap();
        let dists: Vec<f64> = result.distances.to_vec();
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!((dists[1] - 1.0).abs() < 1e-10);
        assert!((dists[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bellman_ford() {
        let (client, device) = setup();
        let graph = make_test_graph(&device);

        let result = client.bellman_ford(&graph, 0).unwrap();
        let dists: Vec<f64> = result.distances.to_vec();
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!((dists[1] - 1.0).abs() < 1e-10);
        assert!((dists[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_floyd_warshall() {
        let (client, device) = setup();
        let graph = make_test_graph(&device);

        let result = client.floyd_warshall(&graph).unwrap();
        let dists: Vec<f64> = result.distances.to_vec();
        // Row 0: [0, 1, 3]
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!((dists[1] - 1.0).abs() < 1e-10);
        assert!((dists[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_astar() {
        let (client, device) = setup();
        let graph = make_test_graph(&device);

        // Heuristic: estimated distance to node 2
        let h = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 0.0], &[3], &device);
        let result = client.astar(&graph, 0, 2, &h).unwrap();
        assert!((result.distance - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_johnson() {
        let (client, device) = setup();
        let graph = make_test_graph(&device);

        let result = client.johnson(&graph).unwrap();
        let dists: Vec<f64> = result.distances.to_vec();
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!((dists[1] - 1.0).abs() < 1e-10);
        assert!((dists[2] - 3.0).abs() < 1e-10);
    }
}
