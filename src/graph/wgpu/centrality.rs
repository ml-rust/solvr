//! WebGPU implementation of centrality algorithms.

use crate::graph::impl_generic::{
    betweenness_centrality_impl, closeness_centrality_impl, degree_centrality_impl,
    eigenvector_centrality_impl, pagerank_impl,
};
use crate::graph::traits::centrality::CentralityAlgorithms;
use crate::graph::traits::types::{EigCentralityOptions, GraphData, PageRankOptions};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CentralityAlgorithms<WgpuRuntime> for WgpuClient {
    fn degree_centrality(&self, graph: &GraphData<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        degree_centrality_impl(self, graph)
    }

    fn betweenness_centrality(
        &self,
        graph: &GraphData<WgpuRuntime>,
        normalized: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        betweenness_centrality_impl(self, graph, normalized)
    }

    fn closeness_centrality(&self, graph: &GraphData<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        closeness_centrality_impl(self, graph)
    }

    fn eigenvector_centrality(
        &self,
        graph: &GraphData<WgpuRuntime>,
        options: &EigCentralityOptions,
    ) -> Result<Tensor<WgpuRuntime>> {
        eigenvector_centrality_impl(self, graph, options)
    }

    fn pagerank(
        &self,
        graph: &GraphData<WgpuRuntime>,
        options: &PageRankOptions,
    ) -> Result<Tensor<WgpuRuntime>> {
        pagerank_impl(self, graph, options)
    }
}
