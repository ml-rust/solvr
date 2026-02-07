//! Eigenvector centrality via power iteration.
//!
//! GPU-parallel via SpMV iteration.

use numr::error::{Error, Result};
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::{SparseOps, SparseTensor};
use numr::tensor::Tensor;

use crate::graph::traits::types::{EigCentralityOptions, GraphData};

/// Eigenvector centrality via power iteration.
///
/// x_{k+1} = A * x_k / ||A * x_k||
/// Converges to the eigenvector of the largest eigenvalue.
pub fn eigenvector_centrality_impl<R, C>(
    client: &C,
    graph: &GraphData<R>,
    options: &EigCentralityOptions,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + SparseOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let n = graph.num_nodes;
    if n == 0 {
        let device = match &graph.adjacency {
            SparseTensor::Csr(csr) => csr.values().device().clone(),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "graph",
                    reason: "Expected CSR format".to_string(),
                });
            }
        };
        return Ok(Tensor::<R>::from_slice(&[] as &[f64], &[0], &device));
    }

    // Initialize x = 1/sqrt(n) for all nodes
    let init_val = 1.0 / (n as f64).sqrt();
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => {
            return Err(Error::InvalidArgument {
                arg: "graph",
                reason: "Expected CSR format".to_string(),
            });
        }
    };

    let mut x = Tensor::<R>::from_slice(&vec![init_val; n], &[n], &device);

    for _ in 0..options.max_iter {
        // x_new = A * x
        let x_new = client.spmv(&graph.adjacency, &x)?;

        // Compute L2 norm: ||x_new|| = sqrt(sum(x_new^2))
        let x_sq = client.mul(&x_new, &x_new)?;
        let sum_sq = client.sum(&x_sq, &[0], false)?;
        let norm_val: f64 = sum_sq.to_vec()[0]; // Single scalar transfer for convergence
        let norm = norm_val.sqrt();

        if norm < 1e-15 {
            // Graph may be disconnected or have no edges
            return Ok(x);
        }

        // Normalize
        let x_normalized = client.mul_scalar(&x_new, 1.0 / norm)?;

        // Check convergence: ||x_new - x||
        let diff = client.sub(&x_normalized, &x)?;
        let diff_sq = client.mul(&diff, &diff)?;
        let diff_sum = client.sum(&diff_sq, &[0], false)?;
        let diff_norm: f64 = diff_sum.to_vec()[0]; // Single scalar transfer

        x = x_normalized;

        if diff_norm.sqrt() < options.tol {
            break;
        }
    }

    // Ensure all values are non-negative (take abs)
    client.abs(&x)
}
