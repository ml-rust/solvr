//! K-Means clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Initialization method for K-Means.
#[derive(Debug, Clone, Default)]
pub enum KMeansInit<R: Runtime> {
    /// K-Means++ initialization (default).
    #[default]
    KMeansPlusPlus,
    /// Random selection from data points.
    Random,
    /// User-provided initial centroids [k, d].
    Points(Tensor<R>),
}

/// Algorithm variant for K-Means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KMeansAlgorithm {
    /// Standard Lloyd's algorithm.
    #[default]
    Lloyd,
    /// Elkan's algorithm with triangle inequality pruning.
    Elkan,
}

/// Options for K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansOptions<R: Runtime> {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Maximum iterations per run.
    pub max_iter: usize,
    /// Convergence tolerance on inertia change.
    pub tol: f64,
    /// Number of random restarts (best result kept).
    pub n_init: usize,
    /// Initialization method.
    pub init: KMeansInit<R>,
    /// Algorithm variant.
    pub algorithm: KMeansAlgorithm,
}

impl<R: Runtime> Default for KMeansOptions<R> {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iter: 300,
            tol: 1e-4,
            n_init: 10,
            init: KMeansInit::KMeansPlusPlus,
            algorithm: KMeansAlgorithm::Lloyd,
        }
    }
}

/// Result of K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult<R: Runtime> {
    /// Cluster centroids [k, d].
    pub centroids: Tensor<R>,
    /// Cluster assignment for each point [n] I64.
    pub labels: Tensor<R>,
    /// Sum of squared distances to nearest centroid (scalar).
    pub inertia: Tensor<R>,
    /// Number of iterations run.
    pub n_iter: usize,
}

/// K-Means clustering algorithms.
pub trait KMeansAlgorithms<R: Runtime> {
    /// Fit K-Means clustering to data [n, d].
    fn kmeans(&self, data: &Tensor<R>, options: &KMeansOptions<R>) -> Result<KMeansResult<R>>;

    /// Predict cluster assignments for new data given centroids.
    fn kmeans_predict(&self, centroids: &Tensor<R>, data: &Tensor<R>) -> Result<Tensor<R>>;
}
