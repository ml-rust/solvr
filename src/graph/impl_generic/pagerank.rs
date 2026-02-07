//! PageRank centrality.
//!
//! Iterative computation using CSR arrays (sequential).

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::{GraphData, PageRankOptions};

use super::helpers::extract_csr_arrays;

/// PageRank centrality.
///
/// r = (1-d)/n + d * M^T * r, where M[i,j] = A[i,j] / out_degree[i].
/// Implemented sequentially using CSR arrays.
pub fn pagerank_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    options: &PageRankOptions,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    if n == 0 {
        return Ok(Tensor::<R>::from_slice(&[] as &[f64], &[0], &device));
    }

    let d = options.damping;
    let base = (1.0 - d) / n as f64;

    // Compute out-degrees
    let mut out_degree = vec![0.0f64; n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            out_degree[i] += values[idx];
        }
    }

    // Initialize ranks
    let mut rank = vec![1.0 / n as f64; n];

    for _ in 0..options.max_iter {
        let mut new_rank = vec![base; n];

        // For each source node i, distribute rank[i]/out_degree[i] to neighbors
        for i in 0..n {
            if out_degree[i] <= 0.0 {
                // Dangling node: distribute evenly to all nodes
                let contrib = d * rank[i] / n as f64;
                for r in new_rank.iter_mut() {
                    *r += contrib;
                }
                continue;
            }

            let start = row_ptrs[i] as usize;
            let end = row_ptrs[i + 1] as usize;
            let contrib = d * rank[i] / out_degree[i];

            for idx in start..end {
                let j = col_indices[idx] as usize;
                let weight = values[idx];
                new_rank[j] += contrib * weight;
            }
        }

        // Check convergence
        let diff: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        rank = new_rank;

        if diff < options.tol {
            break;
        }
    }

    Ok(Tensor::<R>::from_slice(&rank, &[n], &device))
}
