//! Graph algorithms for network analysis.
//!
//! This module provides production-ready graph algorithms including:
//! - Shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall, Johnson, A*)
//! - Minimum spanning tree (Kruskal)
//! - Connectivity (connected components, strongly connected components)
//! - Centrality (degree, betweenness, closeness, eigenvector, PageRank)
//! - Network flow (max flow, min cost flow)
//! - Graph matrices (Laplacian, adjacency, incidence)

mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::*;
