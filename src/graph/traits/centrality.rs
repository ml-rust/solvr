//! Graph centrality algorithm traits.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::types::{EigCentralityOptions, GraphData, PageRankOptions};

/// Graph centrality algorithms.
///
/// Measures the relative importance of nodes in a graph.
pub trait CentralityAlgorithms<R: Runtime> {
    /// Degree centrality: fraction of nodes each node is connected to.
    ///
    /// GPU-parallel via sparse_sum_rows.
    /// Returns [n] tensor with centrality values in [0, 1].
    fn degree_centrality(&self, graph: &GraphData<R>) -> Result<Tensor<R>>;

    /// Betweenness centrality: fraction of shortest paths passing through each node.
    ///
    /// Uses Brandes' algorithm (BFS from each source). Sequential.
    ///
    /// # Arguments
    /// * `normalized` - If true, divide by (n-1)(n-2) for directed or (n-1)(n-2)/2 for undirected.
    fn betweenness_centrality(&self, graph: &GraphData<R>, normalized: bool) -> Result<Tensor<R>>;

    /// Closeness centrality: inverse of mean shortest path distance.
    ///
    /// closeness(v) = (n-1) / sum(d(v, u) for u != v)
    /// Sequential (requires shortest paths from each node).
    fn closeness_centrality(&self, graph: &GraphData<R>) -> Result<Tensor<R>>;

    /// Eigenvector centrality via power iteration.
    ///
    /// GPU-parallel via SpMV iteration. High centrality = connected to other
    /// high-centrality nodes.
    fn eigenvector_centrality(
        &self,
        graph: &GraphData<R>,
        options: &EigCentralityOptions,
    ) -> Result<Tensor<R>>;

    /// PageRank centrality.
    ///
    /// GPU-parallel via SpMV iteration. Models random surfer with damping.
    /// r = (1-d)/n + d * M * r, where M is column-normalized adjacency.
    fn pagerank(&self, graph: &GraphData<R>, options: &PageRankOptions) -> Result<Tensor<R>>;
}
