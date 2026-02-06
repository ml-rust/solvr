//! Spectral clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::kmeans::KMeansResult;

/// Affinity matrix construction method.
#[derive(Debug, Clone)]
pub enum AffinityType {
    /// RBF kernel: exp(-gamma * ||x-y||^2). Gamma auto-estimated if None.
    RBF { gamma: Option<f64> },
    /// k-nearest neighbors graph.
    NearestNeighbors { n_neighbors: usize },
    /// User-provided affinity matrix.
    Precomputed,
}

impl Default for AffinityType {
    fn default() -> Self {
        Self::RBF { gamma: None }
    }
}

/// Graph Laplacian type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LaplacianType {
    /// L = D - W.
    Unnormalized,
    /// L_rw = D^{-1} L (random walk normalized).
    RandomWalk,
    /// L_sym = D^{-1/2} L D^{-1/2}.
    #[default]
    SymmetricNormalized,
}

/// Options for spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralOptions {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Affinity construction method.
    pub affinity: AffinityType,
    /// Laplacian type.
    pub laplacian: LaplacianType,
    /// Number of K-Means restarts on embedding.
    pub n_init: usize,
}

impl Default for SpectralOptions {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            affinity: AffinityType::default(),
            laplacian: LaplacianType::default(),
            n_init: 10,
        }
    }
}

/// Spectral clustering algorithms.
pub trait SpectralClusteringAlgorithms<R: Runtime> {
    /// Run spectral clustering on data [n, d] or precomputed affinity [n, n].
    fn spectral_clustering(
        &self,
        data: &Tensor<R>,
        options: &SpectralOptions,
    ) -> Result<KMeansResult<R>>;
}
