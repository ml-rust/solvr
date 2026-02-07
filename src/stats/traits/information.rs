//! Information theory algorithms.

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Information theory algorithms for tensors.
///
/// Provides entropy, divergence, and mutual information measures.
pub trait InformationTheoryAlgorithms<R: Runtime>: TensorOps<R> {
    /// Shannon entropy of a discrete probability distribution.
    ///
    /// H(X) = -Σ p(x) log(p(x))
    ///
    /// # Arguments
    ///
    /// * `pk` - Probability distribution (1-D tensor, must sum to 1)
    /// * `base` - Logarithm base (e.g., 2.0 for bits, E for nats). If None, uses natural log.
    fn entropy(&self, pk: &Tensor<R>, base: Option<f64>) -> Result<Tensor<R>>;

    /// Differential (continuous) entropy estimate.
    ///
    /// Estimates the entropy of a continuous distribution from samples using
    /// k-nearest neighbor distances.
    ///
    /// # Arguments
    ///
    /// * `x` - Sample data (1-D tensor)
    /// * `k` - Number of nearest neighbors to use (default: 3)
    fn differential_entropy(&self, x: &Tensor<R>, k: usize) -> Result<Tensor<R>>;

    /// Kullback-Leibler divergence.
    ///
    /// D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
    ///
    /// # Arguments
    ///
    /// * `pk` - Reference distribution (1-D tensor, must sum to 1)
    /// * `qk` - Comparison distribution (1-D tensor, must sum to 1, same length as pk)
    /// * `base` - Logarithm base. If None, uses natural log.
    fn kl_divergence(&self, pk: &Tensor<R>, qk: &Tensor<R>, base: Option<f64>)
    -> Result<Tensor<R>>;

    /// Mutual information between two discrete random variables.
    ///
    /// I(X; Y) = H(X) + H(Y) - H(X, Y)
    ///
    /// Estimated from samples using a contingency table.
    ///
    /// # Arguments
    ///
    /// * `x` - Samples from first variable (1-D tensor)
    /// * `y` - Samples from second variable (1-D tensor, same length as x)
    /// * `bins` - Number of bins for histogram estimation
    /// * `base` - Logarithm base. If None, uses natural log.
    fn mutual_information(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        bins: usize,
        base: Option<f64>,
    ) -> Result<Tensor<R>>;
}
