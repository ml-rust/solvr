//! CUDA implementation of shortest path algorithms.

use crate::graph::impl_generic::{
    astar_impl, bellman_ford_impl, dijkstra_impl, floyd_warshall_impl, johnson_impl,
};
use crate::graph::traits::shortest_path::ShortestPathAlgorithms;
use crate::graph::traits::types::{AllPairsResult, GraphData, PathResult, ShortestPathResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ShortestPathAlgorithms<CudaRuntime> for CudaClient {
    fn dijkstra(
        &self,
        graph: &GraphData<CudaRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<CudaRuntime>> {
        dijkstra_impl(self, graph, source)
    }

    fn bellman_ford(
        &self,
        graph: &GraphData<CudaRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<CudaRuntime>> {
        bellman_ford_impl(self, graph, source)
    }

    fn floyd_warshall(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<AllPairsResult<CudaRuntime>> {
        floyd_warshall_impl(self, graph)
    }

    fn johnson(&self, graph: &GraphData<CudaRuntime>) -> Result<AllPairsResult<CudaRuntime>> {
        johnson_impl(self, graph)
    }

    fn astar(
        &self,
        graph: &GraphData<CudaRuntime>,
        source: usize,
        target: usize,
        heuristic: &Tensor<CudaRuntime>,
    ) -> Result<PathResult<CudaRuntime>> {
        astar_impl(self, graph, source, target, heuristic)
    }
}
