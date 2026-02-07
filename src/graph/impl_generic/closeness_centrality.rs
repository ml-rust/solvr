//! Closeness centrality: inverse of mean shortest path distance.
//!
//! Sequential: runs Dijkstra from each node.

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::GraphData;

use super::helpers::extract_csr_arrays;

/// Closeness centrality.
///
/// closeness(v) = (n-1) / sum(d(v, u) for u != v)
/// Uses Dijkstra from each node.
pub fn closeness_centrality_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    if n <= 1 {
        return Ok(Tensor::<R>::from_slice(&vec![0.0f64; n], &[n], &device));
    }

    let mut closeness = vec![0.0f64; n];

    // Run Dijkstra from each source
    for source in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        dist[source] = 0.0;

        let mut heap = std::collections::BinaryHeap::new();
        heap.push(std::cmp::Reverse((0u64, source)));

        while let Some(std::cmp::Reverse((d_bits, u))) = heap.pop() {
            let d = f64::from_bits(d_bits);
            if d > dist[u] {
                continue;
            }

            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            for idx in start..end {
                let v = col_indices[idx] as usize;
                let w = values[idx];
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    heap.push(std::cmp::Reverse((new_dist.to_bits(), v)));
                }
            }
        }

        // Sum of distances to reachable nodes
        let total_dist: f64 = dist.iter().filter(|d| d.is_finite() && **d > 0.0).sum();
        let reachable = dist.iter().filter(|d| d.is_finite() && **d > 0.0).count();

        if reachable > 0 && total_dist > 0.0 {
            closeness[source] = reachable as f64 / total_dist;
        }
    }

    Ok(Tensor::<R>::from_slice(&closeness, &[n], &device))
}
