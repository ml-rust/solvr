//! Graph data types and result structures.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

/// Graph representation using sparse adjacency matrix.
///
/// Wraps a CSR sparse adjacency matrix with metadata for type safety.
/// Weights are stored as values in the sparse matrix.
///
/// # Construction
///
/// ```ignore
/// use solvr::graph::GraphData;
/// use numr::sparse::SparseTensor;
///
/// // From edge list
/// let graph = GraphData::from_edge_list(&rows, &cols, Some(&weights), num_nodes, directed, &device)?;
///
/// // From sparse adjacency matrix
/// let graph = GraphData::new(adjacency, directed);
/// ```
#[derive(Debug, Clone)]
pub struct GraphData<R: Runtime> {
    /// CSR sparse adjacency matrix [n, n], weights as values
    pub adjacency: SparseTensor<R>,
    /// Number of nodes in the graph
    pub num_nodes: usize,
    /// Whether the graph is directed
    pub directed: bool,
}

impl<R: Runtime> GraphData<R> {
    /// Create a graph from a sparse adjacency matrix.
    pub fn new(adjacency: SparseTensor<R>, directed: bool) -> Self {
        let num_nodes = adjacency.nrows();
        Self {
            adjacency,
            num_nodes,
            directed,
        }
    }

    /// Create a graph from an edge list.
    ///
    /// # Arguments
    ///
    /// * `sources` - Source node indices (i64 slice)
    /// * `targets` - Target node indices (i64 slice)
    /// * `weights` - Optional edge weights (f64 slice). If None, uses 1.0 for all edges.
    /// * `num_nodes` - Number of nodes in the graph
    /// * `directed` - Whether the graph is directed
    /// * `device` - Device to create tensors on
    pub fn from_edge_list<T: numr::dtype::Element>(
        sources: &[i64],
        targets: &[i64],
        weights: Option<&[T]>,
        num_nodes: usize,
        directed: bool,
        device: &R::Device,
    ) -> Result<Self> {
        let num_edges = sources.len();

        if directed {
            let vals: Vec<T> = if let Some(w) = weights {
                w.to_vec()
            } else {
                vec![T::one(); num_edges]
            };
            let adjacency = SparseTensor::<R>::from_coo_slices(
                sources,
                targets,
                &vals,
                [num_nodes, num_nodes],
                device,
            )?;
            let adjacency = adjacency.to_csr()?;
            Ok(Self::new(adjacency, directed))
        } else {
            // Undirected: add both directions
            let mut all_sources = Vec::with_capacity(num_edges * 2);
            let mut all_targets = Vec::with_capacity(num_edges * 2);
            let mut all_weights = Vec::with_capacity(num_edges * 2);

            for i in 0..num_edges {
                let w = if let Some(ws) = weights {
                    ws[i]
                } else {
                    T::one()
                };
                all_sources.push(sources[i]);
                all_targets.push(targets[i]);
                all_weights.push(w);
                // Reverse edge
                all_sources.push(targets[i]);
                all_targets.push(sources[i]);
                all_weights.push(w);
            }

            let adjacency = SparseTensor::<R>::from_coo_slices(
                &all_sources,
                &all_targets,
                &all_weights,
                [num_nodes, num_nodes],
                device,
            )?;
            let adjacency = adjacency.to_csr()?;
            Ok(Self::new(adjacency, directed))
        }
    }
}

/// Result of single-source shortest path algorithms.
#[derive(Debug, Clone)]
pub struct ShortestPathResult<R: Runtime> {
    /// Distance from source to each node [n]. Infinity for unreachable nodes.
    pub distances: Tensor<R>,
    /// Predecessor of each node on shortest path [n]. -1 for source/unreachable.
    pub predecessors: Tensor<R>,
}

/// Result of all-pairs shortest path algorithms.
#[derive(Debug, Clone)]
pub struct AllPairsResult<R: Runtime> {
    /// Distance matrix [n, n]. distances[i][j] = shortest path from i to j.
    pub distances: Tensor<R>,
    /// Predecessor matrix [n, n]. predecessors[i][j] = previous node on path from i to j.
    pub predecessors: Tensor<R>,
}

/// Result of a specific path query (source to target).
#[derive(Debug, Clone)]
pub struct PathResult<R: Runtime> {
    /// Total distance from source to target. Infinity if unreachable.
    pub distance: f64,
    /// Node indices along the path [path_len]. Empty if unreachable.
    pub path: Tensor<R>,
}

/// Result of minimum spanning tree algorithms.
#[derive(Debug, Clone)]
pub struct MSTResult<R: Runtime> {
    /// Edge sources in the MST [num_mst_edges].
    pub sources: Tensor<R>,
    /// Edge targets in the MST [num_mst_edges].
    pub targets: Tensor<R>,
    /// Edge weights in the MST [num_mst_edges].
    pub weights: Tensor<R>,
    /// Total weight of the MST.
    pub total_weight: f64,
}

/// Result of connected component algorithms.
#[derive(Debug, Clone)]
pub struct ComponentResult<R: Runtime> {
    /// Component label for each node [n].
    pub labels: Tensor<R>,
    /// Number of connected components.
    pub num_components: usize,
}

/// Result of max-flow algorithms.
#[derive(Debug, Clone)]
pub struct FlowResult<R: Runtime> {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Flow on each edge as a sparse matrix [n, n].
    pub flow: Tensor<R>,
}

/// Options for eigenvector centrality.
#[derive(Debug, Clone)]
pub struct EigCentralityOptions {
    /// Maximum number of iterations for power iteration.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for EigCentralityOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Options for PageRank.
#[derive(Debug, Clone)]
pub struct PageRankOptions {
    /// Damping factor (typically 0.85).
    pub damping: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for PageRankOptions {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Options for min-cost flow.
#[derive(Debug, Clone, Default)]
pub struct MinCostFlowOptions {
    /// Cost per unit flow on each edge (sparse matrix [n, n]).
    /// If None, all costs are 1.
    pub costs: Option<Vec<f64>>,
    /// Maximum flow to push. If None, finds min-cost max flow.
    pub max_flow: Option<f64>,
}
