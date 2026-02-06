//! HDBSCAN clustering trait.

use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Method for extracting clusters from the condensed tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusterSelectionMethod {
    /// Excess of Mass (default).
    #[default]
    EOM,
    /// Leaf clusters.
    Leaf,
}

/// Options for HDBSCAN.
#[derive(Debug, Clone)]
pub struct HdbscanOptions {
    /// Minimum cluster size.
    pub min_cluster_size: usize,
    /// Minimum samples for core distance (defaults to min_cluster_size).
    pub min_samples: Option<usize>,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Cluster extraction method.
    pub cluster_selection_method: ClusterSelectionMethod,
    /// Allow a single cluster result.
    pub allow_single_cluster: bool,
}

impl Default for HdbscanOptions {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
            metric: DistanceMetric::Euclidean,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            allow_single_cluster: false,
        }
    }
}

/// Result of HDBSCAN clustering.
#[derive(Debug, Clone)]
pub struct HdbscanResult<R: Runtime> {
    /// Cluster labels [n] I64, -1 for noise.
    pub labels: Tensor<R>,
    /// Membership strength [n].
    pub probabilities: Tensor<R>,
    /// Persistence of each cluster [n_clusters].
    pub cluster_persistence: Tensor<R>,
}

/// HDBSCAN clustering algorithms.
pub trait HdbscanAlgorithms<R: Runtime> {
    /// Run HDBSCAN on data [n, d].
    fn hdbscan(&self, data: &Tensor<R>, options: &HdbscanOptions) -> Result<HdbscanResult<R>>;
}
