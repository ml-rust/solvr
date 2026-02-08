//! Degree centrality: fraction of nodes each node is connected to.
//!
//! GPU-parallel via SpMV with ones vector.

use numr::error::{Error, Result};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::{SparseOps, SparseTensor};
use numr::tensor::Tensor;

use crate::graph::traits::types::GraphData;

/// Degree centrality: out-degree / (n-1) for each node.
///
/// GPU-parallel via spmv(A, ones).
pub fn degree_centrality_impl<R, C>(client: &C, graph: &GraphData<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + SparseOps<R> + ScalarOps<R>,
{
    let n = graph.num_nodes;
    if n <= 1 {
        let device = match &graph.adjacency {
            SparseTensor::Csr(csr) => csr.values().device().clone(),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "graph",
                    reason: "Expected CSR format".to_string(),
                });
            }
        };
        return Ok(Tensor::<R>::from_slice(&vec![0.0f64; n], &[n], &device));
    }

    // degrees = A * ones (row sums)
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => {
            return Err(Error::InvalidArgument {
                arg: "graph",
                reason: "Expected CSR format".to_string(),
            });
        }
    };

    let ones = Tensor::<R>::from_slice(&vec![1.0f64; n], &[n], &device);
    let degrees = client.spmv(&graph.adjacency, &ones)?;

    // centrality = degrees / (n - 1)
    let scale = 1.0 / (n - 1) as f64;
    client.mul_scalar(&degrees, scale)
}
