//! DBSCAN clustering trait.

use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for DBSCAN.
#[derive(Debug, Clone)]
pub struct DbscanOptions {
    /// Neighborhood radius.
    pub eps: f64,
    /// Minimum points to form a core point.
    pub min_samples: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl Default for DbscanOptions {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DbscanResult<R: Runtime> {
    /// Cluster labels [n] I64, -1 for noise.
    pub labels: Tensor<R>,
    /// Indices of core samples [n_core] I64.
    pub core_sample_indices: Tensor<R>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// DBSCAN clustering algorithms.
pub trait DbscanAlgorithms<R: Runtime> {
    /// Run DBSCAN on data [n, d].
    fn dbscan(&self, data: &Tensor<R>, options: &DbscanOptions) -> Result<DbscanResult<R>>;
}
