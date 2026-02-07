//! Minimum spanning tree algorithm traits.

use numr::error::Result;
use numr::runtime::Runtime;

use super::types::{GraphData, MSTResult};

/// Minimum spanning tree algorithms.
///
/// Finds the subset of edges that connects all nodes with minimum total weight.
/// Only meaningful for undirected graphs.
pub trait MSTAlgorithms<R: Runtime> {
    /// Compute the minimum spanning tree using Kruskal's algorithm.
    ///
    /// Sorts edges by weight and greedily adds edges that don't form cycles
    /// (using union-find). Sequential algorithm.
    ///
    /// # Complexity
    /// O(E log E) for sorting + O(E α(V)) for union-find.
    ///
    /// # Errors
    /// Returns error if the graph is directed.
    fn minimum_spanning_tree(&self, graph: &GraphData<R>) -> Result<MSTResult<R>>;
}
