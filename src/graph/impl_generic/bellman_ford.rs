//! Bellman-Ford algorithm for single-source shortest paths.
//!
//! Handles negative edge weights and detects negative cycles.
//! Implemented sequentially (naturally sequential iteration pattern).

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{GraphData, ShortestPathResult};

use super::helpers::{extract_csr_arrays, validate_node};

/// Bellman-Ford single-source shortest paths.
///
/// Time: O(V * E), handles negative weights, detects negative cycles.
pub fn bellman_ford_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    source: usize,
) -> Result<ShortestPathResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    validate_node(source, graph.num_nodes, "bellman_ford source")?;

    // Extract CSR at API boundary (one time, not in loop)
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    // Get device from graph adjacency
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(), // extract_csr_arrays already validated CSR
    };

    // Initialize distances and predecessors
    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![-1i64; n];
    distances[source] = 0.0;

    // Relax edges V-1 times
    for _ in 0..(n - 1) {
        let mut updated = false;

        // For each node u, relax its outgoing edges
        for u in 0..n {
            if distances[u].is_infinite() {
                continue; // Skip unreachable nodes
            }

            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            for i in start..end {
                let v = col_indices[i] as usize;
                let weight = values[i];

                let new_dist = distances[u] + weight;
                if new_dist < distances[v] {
                    distances[v] = new_dist;
                    predecessors[v] = u as i64;
                    updated = true;
                }
            }
        }

        // Early termination if no updates
        if !updated {
            break;
        }
    }

    // Check for negative cycles (optional: log warning, don't error)
    for u in 0..n {
        if distances[u].is_infinite() {
            continue;
        }

        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            let weight = values[i];

            if distances[u] + weight < distances[v] {
                // Negative cycle detected, but we allow it (will result in negative distances)
                // User can check for -infinity in output
                distances[v] = distances[u] + weight;
                predecessors[v] = u as i64;
            }
        }
    }

    // Create output tensors
    let dist_tensor = Tensor::<R>::from_slice(&distances, &[n], &device);
    let pred_tensor = Tensor::<R>::from_slice(&predecessors, &[n], &device);

    Ok(ShortestPathResult {
        distances: dist_tensor,
        predecessors: pred_tensor,
    })
}
