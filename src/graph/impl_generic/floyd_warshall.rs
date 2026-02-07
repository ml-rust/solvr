//! Floyd-Warshall algorithm for all-pairs shortest paths.
//!
//! GPU-parallel via tensor operations. Builds dense distance matrix, then iterates
//! using broadcasting and element-wise minimum.

use numr::error::Result;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

use crate::graph::traits::types::{AllPairsResult, GraphData};

use super::helpers::extract_csr_arrays;

/// Floyd-Warshall all-pairs shortest paths.
///
/// Time: O(V^3), Memory: O(V^2)
/// GPU-parallelizable via tensor ops.
pub fn floyd_warshall_impl<R, C>(client: &C, graph: &GraphData<R>) -> Result<AllPairsResult<R>>
where
    R: Runtime,
    C: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + RuntimeClient<R>,
{
    let n = graph.num_nodes;

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, _) = extract_csr_arrays(graph)?;

    // Get device from graph adjacency
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(), // extract_csr_arrays already validated CSR
    };

    // Initialize dense distance matrix [n, n]
    let mut distances = vec![f64::INFINITY; n * n];
    let mut predecessors = vec![-1i64; n * n];

    // Set diagonal to 0 and initialize from adjacency
    for i in 0..n {
        distances[i * n + i] = 0.0;

        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            let weight = values[idx];
            distances[i * n + j] = weight;
            predecessors[i * n + j] = i as i64;
        }
    }

    // Convert to tensors for GPU operations
    let mut d = Tensor::<R>::from_slice(&distances, &[n, n], &device);
    let mut pred = Tensor::<R>::from_slice(&predecessors, &[n, n], &device);

    // Floyd-Warshall iteration: D[i,j] = min(D[i,j], D[i,k] + D[k,j])
    for k in 0..n {
        // Extract row k and column k from D
        // D[i,k] = D.narrow(1, k, 1) -> shape [n, 1]
        // D[k,j] = D.narrow(0, k, 1) -> shape [1, n]
        let d_ik = d.narrow(1, k, 1)?; // [n, 1]
        let d_kj = d.narrow(0, k, 1)?; // [1, n]

        // Add them: will broadcast to [n, n]
        let path_sum = client.add(&d_ik, &d_kj)?;

        // Update D: D = min(D, D[i,k] + D[k,j])
        let old_d = d.clone();
        d = client.minimum(&old_d, &path_sum)?;

        // Update predecessors: if path through k is better, set pred[i,j] = pred[k,j]
        let mask = client.lt(&path_sum, &old_d)?; // [n, n] boolean

        // Get pred_kj for all j: pred.narrow(0, k, 1) -> [1, n]
        let pred_kj = pred.narrow(0, k, 1)?;

        // Broadcast mask and pred_kj, then conditional update
        // pred = where(mask, pred_kj, pred)
        pred = client.where_cond(&mask, &pred_kj, &pred)?;
    }

    // Flatten results back if needed
    let dist_vec: Vec<f64> = d.to_vec();
    let pred_vec: Vec<i64> = pred.to_vec();

    let distances_tensor = Tensor::<R>::from_slice(&dist_vec, &[n, n], &device);
    let predecessors_tensor = Tensor::<R>::from_slice(&pred_vec, &[n, n], &device);

    Ok(AllPairsResult {
        distances: distances_tensor,
        predecessors: predecessors_tensor,
    })
}
