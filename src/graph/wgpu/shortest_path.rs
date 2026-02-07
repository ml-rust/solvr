//! WebGPU implementation of shortest path algorithms.

use crate::graph::impl_generic::{
    astar_impl, bellman_ford_impl, dijkstra_impl, floyd_warshall_impl, johnson_impl,
};
use crate::graph::traits::shortest_path::ShortestPathAlgorithms;
use crate::graph::traits::types::{AllPairsResult, GraphData, PathResult, ShortestPathResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ShortestPathAlgorithms<WgpuRuntime> for WgpuClient {
    fn dijkstra(
        &self,
        graph: &GraphData<WgpuRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<WgpuRuntime>> {
        dijkstra_impl(self, graph, source)
    }

    fn bellman_ford(
        &self,
        graph: &GraphData<WgpuRuntime>,
        source: usize,
    ) -> Result<ShortestPathResult<WgpuRuntime>> {
        bellman_ford_impl(self, graph, source)
    }

    fn floyd_warshall(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<AllPairsResult<WgpuRuntime>> {
        floyd_warshall_impl(self, graph)
    }

    fn johnson(&self, graph: &GraphData<WgpuRuntime>) -> Result<AllPairsResult<WgpuRuntime>> {
        johnson_impl(self, graph)
    }

    fn astar(
        &self,
        graph: &GraphData<WgpuRuntime>,
        source: usize,
        target: usize,
        heuristic: &Tensor<WgpuRuntime>,
    ) -> Result<PathResult<WgpuRuntime>> {
        astar_impl(self, graph, source, target, heuristic)
    }
}
