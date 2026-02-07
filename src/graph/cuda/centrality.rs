//! CUDA implementation of centrality algorithms.

use crate::graph::impl_generic::{
    betweenness_centrality_impl, closeness_centrality_impl, degree_centrality_impl,
    eigenvector_centrality_impl, pagerank_impl,
};
use crate::graph::traits::centrality::CentralityAlgorithms;
use crate::graph::traits::types::{EigCentralityOptions, GraphData, PageRankOptions};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CentralityAlgorithms<CudaRuntime> for CudaClient {
    fn degree_centrality(&self, graph: &GraphData<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        degree_centrality_impl(self, graph)
    }

    fn betweenness_centrality(
        &self,
        graph: &GraphData<CudaRuntime>,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        betweenness_centrality_impl(self, graph, normalized)
    }

    fn closeness_centrality(&self, graph: &GraphData<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        closeness_centrality_impl(self, graph)
    }

    fn eigenvector_centrality(
        &self,
        graph: &GraphData<CudaRuntime>,
        options: &EigCentralityOptions,
    ) -> Result<Tensor<CudaRuntime>> {
        eigenvector_centrality_impl(self, graph, options)
    }

    fn pagerank(
        &self,
        graph: &GraphData<CudaRuntime>,
        options: &PageRankOptions,
    ) -> Result<Tensor<CudaRuntime>> {
        pagerank_impl(self, graph, options)
    }
}
