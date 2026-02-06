//! Mini-Batch K-Means clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::kmeans::{KMeansInit, KMeansResult};

/// Options for Mini-Batch K-Means.
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansOptions<R: Runtime> {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance (0.0 disables).
    pub tol: f64,
    /// Initialization method.
    pub init: KMeansInit<R>,
    /// Early stopping after this many iterations without improvement.
    pub max_no_improvement: usize,
}

impl<R: Runtime> Default for MiniBatchKMeansOptions<R> {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            batch_size: 1024,
            max_iter: 100,
            tol: 0.0,
            init: KMeansInit::KMeansPlusPlus,
            max_no_improvement: 10,
        }
    }
}

/// Mini-Batch K-Means clustering algorithms.
pub trait MiniBatchKMeansAlgorithms<R: Runtime> {
    /// Fit Mini-Batch K-Means to data [n, d].
    fn mini_batch_kmeans(
        &self,
        data: &Tensor<R>,
        options: &MiniBatchKMeansOptions<R>,
    ) -> Result<KMeansResult<R>>;
}
