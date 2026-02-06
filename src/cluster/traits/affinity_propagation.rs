//! Affinity Propagation clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for Affinity Propagation.
#[derive(Debug, Clone)]
pub struct AffinityPropagationOptions {
    /// Damping factor in [0.5, 1.0).
    pub damping: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Number of iterations with no change for convergence.
    pub convergence_iter: usize,
    /// Preference (self-similarity). None = median of similarities.
    pub preference: Option<f64>,
}

impl Default for AffinityPropagationOptions {
    fn default() -> Self {
        Self {
            damping: 0.5,
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
        }
    }
}

/// Result of Affinity Propagation.
#[derive(Debug, Clone)]
pub struct AffinityPropagationResult<R: Runtime> {
    /// Cluster labels [n] I64.
    pub labels: Tensor<R>,
    /// Indices of exemplars [k] I64.
    pub cluster_centers_indices: Tensor<R>,
    /// Number of iterations run.
    pub n_iter: usize,
}

/// Affinity Propagation clustering algorithms.
pub trait AffinityPropagationAlgorithms<R: Runtime> {
    /// Run Affinity Propagation on a similarity matrix [n, n].
    fn affinity_propagation(
        &self,
        similarities: &Tensor<R>,
        options: &AffinityPropagationOptions,
    ) -> Result<AffinityPropagationResult<R>>;
}
