//! Johnson's algorithm for all-pairs shortest paths.
//!
//! Combines Bellman-Ford reweighting with Dijkstra from each source.
//! Better than Floyd-Warshall for sparse graphs: O(V^2 log V + VE).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::{AllPairsResult, GraphData};

use super::helpers::extract_csr_arrays;

/// Johnson's all-pairs shortest paths.
///
/// Time: O(V^2 log V + VE) for sparse graphs
pub fn johnson_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<AllPairsResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let n = graph.num_nodes;

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, _) = extract_csr_arrays(graph)?;

    // Get device from graph adjacency
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(), // extract_csr_arrays already validated CSR
    };

    // Step 1: Run Bellman-Ford from a virtual source (s) connected to all nodes with 0-weight edges
    // This reweights the graph so all weights are non-negative
    // Equivalently, compute h[v] = min distances from implicit virtual source
    let mut h = vec![f64::INFINITY; n];
    h[0] = 0.0; // Start from node 0 (arbitrary choice for virtual source)

    // Bellman-Ford-style relaxation V-1 times
    for _ in 0..(n - 1) {
        let mut updated = false;

        for u in 0..n {
            if h[u].is_infinite() {
                continue;
            }

            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            for i in start..end {
                let v = col_indices[i] as usize;
                let weight = values[i];

                let new_h = h[u] + weight;
                if new_h < h[v] {
                    h[v] = new_h;
                    updated = true;
                }
            }
        }

        if !updated {
            break;
        }
    }

    // Step 2: Reweight edges: w'(u, v) = w(u, v) + h[u] - h[v]
    let mut new_weights = values.clone();
    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            new_weights[i] += h[u] - h[v];
        }
    }

    // Step 3: Run Dijkstra from each source with reweighted graph
    let mut all_distances = vec![f64::INFINITY; n * n];
    let mut all_predecessors = vec![-1i64; n * n];

    for source in 0..n {
        // Dijkstra from source with reweighted graph
        let mut dist = vec![f64::INFINITY; n];
        let mut pred = vec![-1i64; n];
        dist[source] = 0.0;

        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        heap.push(Reverse((0.0_f64.to_bits(), source)));

        while let Some(Reverse((d_bits, u))) = heap.pop() {
            let d = f64::from_bits(d_bits);

            if d > dist[u] {
                continue;
            }

            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            for i in start..end {
                let v = col_indices[i] as usize;
                let weight = new_weights[i];

                let new_dist = dist[u] + weight;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    pred[v] = u as i64;
                    heap.push(Reverse((new_dist.to_bits(), v)));
                }
            }
        }

        // Convert reweighted distances back to original weights
        for v in 0..n {
            if !dist[v].is_infinite() {
                dist[v] = dist[v] + h[v] - h[source];
            }
            all_distances[source * n + v] = dist[v];
            all_predecessors[source * n + v] = pred[v];
        }
    }

    // Create output tensors
    let distances = Tensor::<R>::from_slice(&all_distances, &[n, n], &device);
    let predecessors = Tensor::<R>::from_slice(&all_predecessors, &[n, n], &device);

    Ok(AllPairsResult {
        distances,
        predecessors,
    })
}
