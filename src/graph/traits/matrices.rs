//! Graph matrix construction traits.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::SparseTensor;

use super::types::GraphData;

/// Graph matrix construction algorithms.
///
/// Builds standard graph matrices (Laplacian, incidence) from adjacency data.
pub trait GraphMatrixAlgorithms<R: Runtime> {
    /// Compute the graph Laplacian matrix.
    ///
    /// L = D - A, where D is the degree matrix and A is the adjacency matrix.
    /// If normalized: L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}.
    ///
    /// GPU-parallel via sparse operations.
    fn laplacian_matrix(&self, graph: &GraphData<R>, normalized: bool) -> Result<SparseTensor<R>>;

    /// Return the adjacency matrix (possibly symmetrized for undirected graphs).
    fn adjacency_matrix(&self, graph: &GraphData<R>) -> Result<SparseTensor<R>>;

    /// Compute the incidence matrix.
    ///
    /// B[i, e] = -1 if edge e leaves node i, +1 if edge e enters node i.
    /// For undirected graphs, arbitrary orientation is assigned.
    fn incidence_matrix(&self, graph: &GraphData<R>) -> Result<SparseTensor<R>>;
}
