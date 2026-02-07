//! Maximum flow via Edmonds-Karp (BFS-based Ford-Fulkerson).
//!
//! Implemented sequentially at API boundary.

use std::collections::VecDeque;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{FlowResult, GraphData};

use super::helpers::{extract_csr_arrays, validate_node};

/// Maximum flow via Edmonds-Karp (BFS-based Ford-Fulkerson).
///
/// Time: O(V * E^2).
/// Returns flow as a dense [n, n] matrix (flattened).
pub fn max_flow_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    source: usize,
    sink: usize,
) -> Result<FlowResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    validate_node(source, graph.num_nodes, "max_flow source")?;
    validate_node(sink, graph.num_nodes, "max_flow sink")?;

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    // Get device from graph
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Build residual graph as dense adjacency matrix (capacity = weight)
    let mut capacity = vec![vec![0.0; n]; n];
    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            capacity[u][v] += values[i]; // Sum in case of duplicates
        }
    }

    // Flow matrix (initially zero)
    let mut flow = vec![vec![0.0; n]; n];
    let mut max_flow = 0.0;

    // BFS to find augmenting path
    loop {
        // BFS from source to sink in residual graph
        let mut parent = vec![-1i64; n];
        let mut queue = VecDeque::new();
        queue.push_back(source);
        parent[source] = source as i64;

        while let Some(u) = queue.pop_front() {
            if u == sink {
                break;
            }

            // Find neighbors with available capacity
            for v in 0..n {
                let residual_capacity = capacity[u][v] - flow[u][v] + flow[v][u];
                if parent[v] == -1 && residual_capacity > 0.0 {
                    parent[v] = u as i64;
                    queue.push_back(v);
                }
            }
        }

        // If no path found, we're done
        if parent[sink] == -1 {
            break;
        }

        // Find bottleneck capacity along the path
        let mut path_flow = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v] as usize;
            let residual_capacity = capacity[u][v] - flow[u][v] + flow[v][u];
            path_flow = path_flow.min(residual_capacity);
            v = u;
        }

        // Update flow along the path
        v = sink;
        while v != source {
            let u = parent[v] as usize;
            flow[u][v] += path_flow;
            flow[v][u] -= path_flow; // Reverse edge
            v = u;
        }

        max_flow += path_flow;
    }

    // Create output tensor: flatten [n, n] -> [n*n]
    let mut flow_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            flow_flat.push(flow[i][j]);
        }
    }

    let flow_tensor = Tensor::<R>::from_slice(&flow_flat, &[n * n], &device);

    Ok(FlowResult {
        max_flow,
        flow: flow_tensor,
    })
}
