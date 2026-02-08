//! Shared helper functions for graph algorithms.

use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::sparse::SparseTensor;

use crate::graph::traits::types::GraphData;

pub type CsrArrays = (Vec<i64>, Vec<i64>, Vec<f64>, usize);

/// Extract CSR arrays from a GraphData for sequential algorithms.
///
/// Returns (row_ptrs, col_indices, values, num_nodes) as CPU vecs.
/// This is only used at API boundary for inherently sequential algorithms
/// (Dijkstra, A*, Tarjan, Kruskal, Ford-Fulkerson).
pub fn extract_csr_arrays<R: Runtime>(graph: &GraphData<R>) -> Result<CsrArrays> {
    let csr = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr,
        _ => {
            return Err(Error::InvalidArgument {
                arg: "graph",
                reason: "Graph adjacency must be in CSR format. Call to_csr() first.".to_string(),
            });
        }
    };

    let row_ptrs: Vec<i64> = csr.row_ptrs().to_vec();
    let col_indices: Vec<i64> = csr.col_indices().to_vec();
    let values: Vec<f64> = csr.values().to_vec();
    let n = graph.num_nodes;

    Ok((row_ptrs, col_indices, values, n))
}

/// Validate that source node is within bounds.
pub fn validate_node(node: usize, num_nodes: usize, name: &str) -> Result<()> {
    if node >= num_nodes {
        return Err(Error::InvalidArgument {
            arg: "node",
            reason: format!("{name} node {node} >= num_nodes {num_nodes}"),
        });
    }
    Ok(())
}
