//! Minimum cost maximum flow via successive shortest path algorithm.
//!
//! Implemented sequentially at API boundary.

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{FlowResult, GraphData, MinCostFlowOptions};

use super::helpers::{extract_csr_arrays, validate_node};

/// Minimum cost flow via successive shortest path algorithm.
///
/// Time: O(V * E * flow_value) using Bellman-Ford for shortest paths.
/// Returns flow as a dense [n, n] matrix (flattened).
pub fn min_cost_flow_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    source: usize,
    sink: usize,
    options: &MinCostFlowOptions,
) -> Result<FlowResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    validate_node(source, graph.num_nodes, "min_cost_flow source")?;
    validate_node(sink, graph.num_nodes, "min_cost_flow sink")?;

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    // Get device from graph
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Build capacity matrix from adjacency
    let mut capacity = vec![vec![0.0; n]; n];
    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            capacity[u][v] += values[i];
        }
    }

    // Build cost matrix
    let mut cost = vec![vec![0.0; n]; n];
    if let Some(costs) = &options.costs {
        // Assume costs is provided as a flat [n*n] array
        if costs.len() == n * n {
            for i in 0..n {
                for j in 0..n {
                    cost[i][j] = costs[i * n + j];
                }
            }
        }
    }

    // Flow matrix
    let mut flow = vec![vec![0.0; n]; n];
    let mut total_cost = 0.0;
    let mut total_flow = 0.0;

    // Successive shortest path: repeatedly find min-cost augmenting path
    loop {
        // Check if we've reached max_flow limit
        if let Some(max_flow_limit) = options.max_flow {
            if total_flow >= max_flow_limit {
                break;
            }
        }

        // Bellman-Ford to find shortest path (allowing negative costs due to backward edges)
        let mut dist = vec![f64::INFINITY; n];
        let mut parent = vec![-1i64; n];
        dist[source] = 0.0;

        // Relax edges n-1 times
        for _ in 0..(n - 1) {
            for u in 0..n {
                if dist[u].is_infinite() {
                    continue;
                }

                for v in 0..n {
                    let edge_cost = if capacity[u][v] > flow[u][v] {
                        cost[u][v] // Forward edge
                    } else if flow[u][v] > 0.0 {
                        -cost[u][v] // Backward edge (reverse flow)
                    } else {
                        continue; // No edge
                    };

                    let new_dist = dist[u] + edge_cost;
                    if new_dist < dist[v] {
                        dist[v] = new_dist;
                        parent[v] = u as i64;
                    }
                }
            }
        }

        // If no path found, we're done
        if parent[sink] == -1 {
            break;
        }

        // Find bottleneck capacity along the path
        let mut path_flow = match options.max_flow {
            Some(limit) => (limit - total_flow).min(f64::INFINITY),
            None => f64::INFINITY,
        };

        let mut v = sink;
        while v != source {
            let u = parent[v] as usize;
            if capacity[u][v] > flow[u][v] {
                path_flow = path_flow.min(capacity[u][v] - flow[u][v]);
            } else if flow[u][v] > 0.0 {
                path_flow = path_flow.min(flow[u][v]);
            }
            v = u;
        }

        if path_flow.is_infinite() || path_flow <= 0.0 {
            break;
        }

        // Update flow along the path
        let mut path_cost = 0.0;
        v = sink;
        while v != source {
            let u = parent[v] as usize;
            if capacity[u][v] > flow[u][v] {
                // Forward edge
                flow[u][v] += path_flow;
                path_cost += path_flow * cost[u][v];
            } else {
                // Backward edge (decrease reverse flow)
                flow[u][v] -= path_flow;
                path_cost -= path_flow * cost[u][v];
            }
            v = u;
        }

        total_flow += path_flow;
        total_cost += path_cost;
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
        max_flow: total_cost, // Return cost as "max_flow" (API uses this for cost value)
        flow: flow_tensor,
    })
}
