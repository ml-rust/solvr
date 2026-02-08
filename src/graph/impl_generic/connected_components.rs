//! Connected components via BFS (undirected graphs).
//!
//! Labels each node with its component ID (smallest node index in component).
//! Treats directed graphs as undirected (ignores edge direction).
//! Implemented sequentially at API boundary.

use std::collections::VecDeque;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{ComponentResult, GraphData};

use super::helpers::extract_csr_arrays;

/// Connected components via BFS.
///
/// Time: O(V + E). Each node gets labeled with its component ID
/// (smallest node index in the component).
pub fn connected_components_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
) -> Result<ComponentResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    // Extract CSR at API boundary
    let (row_ptrs, col_indices, _values, n) = extract_csr_arrays(graph)?;

    // Get device from graph
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    let mut labels = vec![-1i64; n];
    let mut num_components = 0;

    // Build adjacency list from CSR
    // For undirected graphs, GraphData already stores both directions,
    // so we just read the CSR as-is.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for &v_idx in col_indices.iter().take(end).skip(start) {
            let v = v_idx as usize;
            adj[u].push(v);
        }
    }

    // BFS from each unvisited node
    for start_node in 0..n {
        if labels[start_node] != -1 {
            continue; // Already labeled
        }

        let component_id = start_node as i64;
        let mut queue = VecDeque::new();
        queue.push_back(start_node);
        labels[start_node] = component_id;

        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if labels[v] == -1 {
                    labels[v] = component_id;
                    queue.push_back(v);
                }
            }
        }

        num_components += 1;
    }

    // Create output tensor
    let labels_tensor = Tensor::<R>::from_slice(&labels, &[n], &device);

    Ok(ComponentResult {
        labels: labels_tensor,
        num_components,
    })
}
