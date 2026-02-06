//! Bisecting K-Means clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::kmeans::KMeansResult;

/// Strategy for selecting which cluster to bisect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BisectingStrategy {
    /// Bisect the cluster with the most points.
    #[default]
    BiggestCluster,
    /// Bisect the cluster with highest SSE.
    HighestSSE,
}

/// Options for Bisecting K-Means.
#[derive(Debug, Clone)]
pub struct BisectingKMeansOptions {
    /// Target number of clusters.
    pub n_clusters: usize,
    /// Max iterations per bisection step.
    pub max_iter: usize,
    /// Convergence tolerance per bisection.
    pub tol: f64,
    /// Random restarts per bisection.
    pub n_init: usize,
    /// Strategy for selecting cluster to bisect.
    pub bisecting_strategy: BisectingStrategy,
}

impl Default for BisectingKMeansOptions {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iter: 300,
            tol: 1e-4,
            n_init: 1,
            bisecting_strategy: BisectingStrategy::BiggestCluster,
        }
    }
}

/// Bisecting K-Means clustering algorithms.
pub trait BisectingKMeansAlgorithms<R: Runtime> {
    /// Fit Bisecting K-Means to data [n, d].
    fn bisecting_kmeans(
        &self,
        data: &Tensor<R>,
        options: &BisectingKMeansOptions,
    ) -> Result<KMeansResult<R>>;
}
