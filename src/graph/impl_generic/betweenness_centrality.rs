//! Betweenness centrality via Brandes' algorithm.
//!
//! Sequential BFS from each source, accumulates pair dependencies.

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::GraphData;

use super::helpers::extract_csr_arrays;

/// Betweenness centrality via Brandes' algorithm.
///
/// Time: O(V * E) for unweighted graphs.
pub fn betweenness_centrality_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    normalized: bool,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let (row_ptrs, col_indices, _values, n) = extract_csr_arrays(graph)?;

    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    if n == 0 {
        return Ok(Tensor::<R>::from_slice(&[] as &[f64], &[0], &device));
    }

    let mut betweenness = vec![0.0f64; n];

    // Brandes' algorithm: BFS from each source
    for s in 0..n {
        let mut stack = Vec::with_capacity(n);
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n]; // number of shortest paths
        let mut dist = vec![-1i64; n];
        let mut delta = vec![0.0f64; n];

        sigma[s] = 1.0;
        dist[s] = 0;

        // BFS
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);

            let start = row_ptrs[v] as usize;
            let end = row_ptrs[v + 1] as usize;
            for idx in start..end {
                let w = col_indices[idx] as usize;

                // First visit?
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }

                // Shortest path to w through v?
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation of dependencies
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                betweenness[w] += delta[w];
            }
            delta[w] = 0.0;
        }
    }

    // For undirected graphs, each pair is counted twice
    if !graph.directed {
        for b in betweenness.iter_mut() {
            *b /= 2.0;
        }
    }

    // Normalize
    if normalized && n > 2 {
        let norm = if graph.directed {
            ((n - 1) * (n - 2)) as f64
        } else {
            ((n - 1) * (n - 2)) as f64 / 2.0
        };
        for b in betweenness.iter_mut() {
            *b /= norm;
        }
    }

    Ok(Tensor::<R>::from_slice(&betweenness, &[n], &device))
}
