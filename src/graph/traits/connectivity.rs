//! Graph connectivity algorithm traits.

use numr::error::Result;
use numr::runtime::Runtime;

use super::types::{ComponentResult, GraphData};

/// Graph connectivity algorithms.
///
/// Determines connected components and connectivity properties.
pub trait ConnectivityAlgorithms<R: Runtime> {
    /// Find connected components of an undirected graph.
    ///
    /// Uses BFS-based label propagation. Each node gets labeled with its
    /// component ID (smallest node index in the component).
    ///
    /// For directed graphs, treats edges as undirected.
    fn connected_components(&self, graph: &GraphData<R>) -> Result<ComponentResult<R>>;

    /// Find strongly connected components of a directed graph.
    ///
    /// Uses Tarjan's algorithm (sequential DFS-based).
    /// For undirected graphs, equivalent to connected_components.
    fn strongly_connected_components(&self, graph: &GraphData<R>) -> Result<ComponentResult<R>>;

    /// Check if the graph is connected (undirected) or weakly connected (directed).
    fn is_connected(&self, graph: &GraphData<R>) -> Result<bool>;

    /// Check if the directed graph is strongly connected.
    ///
    /// Every node is reachable from every other node.
    fn is_strongly_connected(&self, graph: &GraphData<R>) -> Result<bool>;
}
