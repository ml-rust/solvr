//! A* search for shortest path from source to target.
//!
//! Uses heuristic function to guide search with priority queue.
//! Implemented sequentially at API boundary.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::{GraphData, PathResult};

use super::helpers::{extract_csr_arrays, validate_node};

/// A* search for shortest path from source to target.
///
/// Uses heuristic to guide search. Heuristic must be admissible (not overestimate).
pub fn astar_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    source: usize,
    target: usize,
    heuristic: &Tensor<R>,
) -> Result<PathResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    validate_node(source, graph.num_nodes, "astar source")?;
    validate_node(target, graph.num_nodes, "astar target")?;

    if heuristic.shape().len() != 1 || heuristic.shape()[0] != graph.num_nodes {
        return Err(numr::error::Error::InvalidArgument {
            arg: "heuristic",
            reason: format!(
                "heuristic must be [{}], got {:?}",
                graph.num_nodes,
                heuristic.shape()
            ),
        });
    }

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    // Extract heuristic values to CPU
    let h_vals: Vec<f64> = heuristic.to_vec();

    // Get device from graph adjacency
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(), // extract_csr_arrays already validated CSR
    };

    // A* data structures
    let mut g_score = vec![f64::INFINITY; n]; // Cost from source
    let mut parent = vec![-1i64; n];
    g_score[source] = 0.0;

    // Priority queue: (f_score, node_id) where f = g + h
    let mut heap = BinaryHeap::new();
    let f_init = h_vals[source];
    heap.push(Reverse((f_init.to_bits() as i64, source as i64))); // Convert f64 to bits for ordering

    while let Some(Reverse((f_bits, current_i64))) = heap.pop() {
        let current = current_i64 as usize;
        let f_score = f64::from_bits(f_bits as u64);

        // Early termination if target reached
        if current == target {
            // Reconstruct path
            let mut path_vec = Vec::new();
            let mut node = target;
            while node != source && parent[node] >= 0 {
                path_vec.push(node as i64);
                node = parent[node] as usize;
            }
            path_vec.push(source as i64);
            path_vec.reverse();

            let distance = g_score[target];
            let path = Tensor::<R>::from_slice(&path_vec, &[path_vec.len()], &device);

            return Ok(PathResult { distance, path });
        }

        // Skip if we've found a better path to current
        let g_current = g_score[current];
        if g_current + h_vals[current] < f_score {
            continue;
        }

        // Explore neighbors
        let start = row_ptrs[current] as usize;
        let end = row_ptrs[current + 1] as usize;
        for i in start..end {
            let neighbor = col_indices[i] as usize;
            let weight = values[i];

            let new_g = g_current + weight;
            if new_g < g_score[neighbor] {
                g_score[neighbor] = new_g;
                parent[neighbor] = current as i64;

                let f_new = new_g + h_vals[neighbor];
                heap.push(Reverse((f_new.to_bits() as i64, neighbor as i64)));
            }
        }
    }

    // Target not reached
    let empty_path = Tensor::<R>::from_slice(&[] as &[i64], &[0], &device);
    Ok(PathResult {
        distance: f64::INFINITY,
        path: empty_path,
    })
}
