//! Shortest path algorithm traits.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::types::{AllPairsResult, GraphData, PathResult, ShortestPathResult};

/// Shortest path algorithms for graphs.
///
/// Provides single-source (Dijkstra, Bellman-Ford), all-pairs (Floyd-Warshall, Johnson),
/// and heuristic (A*) shortest path computation.
pub trait ShortestPathAlgorithms<R: Runtime> {
    /// Dijkstra's algorithm for single-source shortest paths.
    ///
    /// Requires non-negative edge weights. Uses priority queue (sequential).
    ///
    /// # Complexity
    /// O((V + E) log V) with binary heap.
    fn dijkstra(&self, graph: &GraphData<R>, source: usize) -> Result<ShortestPathResult<R>>;

    /// Bellman-Ford algorithm for single-source shortest paths.
    ///
    /// Handles negative edge weights. Detects negative cycles.
    /// GPU-parallel via scatter_reduce edge relaxation.
    ///
    /// # Complexity
    /// O(V * E), parallelizable across edges.
    fn bellman_ford(&self, graph: &GraphData<R>, source: usize) -> Result<ShortestPathResult<R>>;

    /// Floyd-Warshall algorithm for all-pairs shortest paths.
    ///
    /// GPU-parallel via matrix operations. Uses dense distance matrix.
    ///
    /// # Complexity
    /// O(V^3) compute, O(V^2) memory.
    fn floyd_warshall(&self, graph: &GraphData<R>) -> Result<AllPairsResult<R>>;

    /// Johnson's algorithm for all-pairs shortest paths.
    ///
    /// Combines Bellman-Ford reweighting with Dijkstra from each source.
    /// Better than Floyd-Warshall for sparse graphs.
    ///
    /// # Complexity
    /// O(V^2 log V + VE).
    fn johnson(&self, graph: &GraphData<R>) -> Result<AllPairsResult<R>>;

    /// A* search for shortest path from source to target.
    ///
    /// Uses heuristic function for guided search. Sequential (priority queue).
    ///
    /// # Arguments
    ///
    /// * `heuristic` - Estimated distance from each node to target [n].
    ///   Must be admissible (never overestimates).
    fn astar(
        &self,
        graph: &GraphData<R>,
        source: usize,
        target: usize,
        heuristic: &Tensor<R>,
    ) -> Result<PathResult<R>>;
}
