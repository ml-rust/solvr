//! Minimum spanning tree via Kruskal's algorithm (undirected graphs).
//!
//! Uses union-find with path compression and union by rank.
//! Implemented sequentially at API boundary.

use numr::error::{Error, Result};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{GraphData, MSTResult};

use super::helpers::extract_csr_arrays;

/// Union-Find data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let px = self.find(x);
        let py = self.find(y);

        if px == py {
            return false; // Already in same set
        }

        // Union by rank
        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }

        true
    }
}

/// Kruskal's minimum spanning tree algorithm.
///
/// Time: O(E log E) for sorting + O(E α(V)) for union-find.
pub fn kruskal_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<MSTResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if graph.directed {
        return Err(Error::InvalidArgument {
            arg: "graph",
            reason: "Kruskal's MST only works on undirected graphs".to_string(),
        });
    }

    // Extract CSR at API boundary
    let (row_ptrs, col_indices, values, n) = extract_csr_arrays(graph)?;

    // Get device from graph
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Collect all edges from CSR (avoiding duplicates in undirected graph)
    // For undirected graph stored as CSR, we take only edges where source < target
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();

    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            let weight = values[i];

            // Only add edge if u < v to avoid duplicates
            if u < v {
                edges.push((u, v, weight));
            }
        }
    }

    // Sort edges by weight (ascending)
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Kruskal's: greedily add edges that don't form cycles
    let mut uf = UnionFind::new(n);
    let mut mst_sources = Vec::new();
    let mut mst_targets = Vec::new();
    let mut mst_weights = Vec::new();
    let mut total_weight = 0.0;

    for (u, v, weight) in edges {
        if uf.union(u, v) {
            mst_sources.push(u as i64);
            mst_targets.push(v as i64);
            mst_weights.push(weight);
            total_weight += weight;

            // Stop when we have n-1 edges
            if mst_sources.len() == n - 1 {
                break;
            }
        }
    }

    // Create output tensors
    let sources = if mst_sources.is_empty() {
        Tensor::<R>::from_slice(&[0i64; 0], &[0], &device)
    } else {
        Tensor::<R>::from_slice(&mst_sources, &[mst_sources.len()], &device)
    };

    let targets = if mst_targets.is_empty() {
        Tensor::<R>::from_slice(&[0i64; 0], &[0], &device)
    } else {
        Tensor::<R>::from_slice(&mst_targets, &[mst_targets.len()], &device)
    };

    let weights = if mst_weights.is_empty() {
        Tensor::<R>::from_slice(&[0.0f64; 0], &[0], &device)
    } else {
        Tensor::<R>::from_slice(&mst_weights, &[mst_weights.len()], &device)
    };

    Ok(MSTResult {
        sources,
        targets,
        weights,
        total_weight,
    })
}
