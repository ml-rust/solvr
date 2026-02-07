//! Dijkstra's algorithm for single-source shortest paths.
//!
//! Uses a binary heap-based priority queue. Requires non-negative edge weights.
//! Implemented sequentially at API boundary (inherently sequential algorithm).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{GraphData, ShortestPathResult};

use super::helpers::{extract_csr_arrays, validate_node};

/// Dijkstra's single-source shortest paths via binary heap.
///
/// Time: O((V + E) log V)
pub fn dijkstra_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    source: usize,
) -> Result<ShortestPathResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    validate_node(source, graph.num_nodes, "dijkstra source")?;

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

    // Priority queue: (distance_bits, node)
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0.0_f64.to_bits(), source as i64)));

    while let Some(Reverse((dist_bits, u))) = heap.pop() {
        let u = u as usize;
        let dist = f64::from_bits(dist_bits);

        // Skip if we've already found a better path
        if dist > distances[u] {
            continue;
        }

        // Relax edges from u
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            let weight = values[i];

            let new_dist = distances[u] + weight;
            if new_dist < distances[v] {
                distances[v] = new_dist;
                predecessors[v] = u as i64;
                heap.push(Reverse((new_dist.to_bits(), v as i64)));
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
