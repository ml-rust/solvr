//! OPTICS clustering trait.

use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for OPTICS.
#[derive(Debug, Clone)]
pub struct OpticsOptions {
    /// Minimum samples for core distance.
    pub min_samples: usize,
    /// Maximum neighborhood radius.
    pub max_eps: f64,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Xi parameter for cluster extraction (None = no extraction).
    pub xi: Option<f64>,
}

impl Default for OpticsOptions {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_eps: f64::INFINITY,
            metric: DistanceMetric::Euclidean,
            xi: None,
        }
    }
}

/// Result of OPTICS.
#[derive(Debug, Clone)]
pub struct OpticsResult<R: Runtime> {
    /// Processing order [n] I64.
    pub ordering: Tensor<R>,
    /// Reachability distances [n].
    pub reachability: Tensor<R>,
    /// Core distances [n].
    pub core_distances: Tensor<R>,
    /// Cluster labels [n] I64 (if xi provided).
    pub labels: Tensor<R>,
}

/// OPTICS clustering algorithms.
pub trait OpticsAlgorithms<R: Runtime> {
    /// Run OPTICS on data [n, d].
    fn optics(&self, data: &Tensor<R>, options: &OpticsOptions) -> Result<OpticsResult<R>>;
}
