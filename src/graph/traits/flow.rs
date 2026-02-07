//! Network flow algorithm traits.

use numr::error::Result;
use numr::runtime::Runtime;

use super::types::{FlowResult, GraphData, MinCostFlowOptions};

/// Network flow algorithms.
///
/// Computes maximum flow and minimum cost flow in directed graphs.
pub trait FlowAlgorithms<R: Runtime> {
    /// Maximum flow via Edmonds-Karp (BFS-based Ford-Fulkerson).
    ///
    /// Sequential (BFS augmenting paths).
    ///
    /// # Complexity
    /// O(V * E^2).
    fn max_flow(&self, graph: &GraphData<R>, source: usize, sink: usize) -> Result<FlowResult<R>>;

    /// Minimum cost maximum flow.
    ///
    /// Sequential (successive shortest path algorithm).
    ///
    /// # Complexity
    /// O(V * E * max_flow).
    fn min_cost_flow(
        &self,
        graph: &GraphData<R>,
        source: usize,
        sink: usize,
        options: &MinCostFlowOptions,
    ) -> Result<FlowResult<R>>;
}
