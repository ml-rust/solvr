//! Information theory algorithms.
use crate::DType;

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Information theory algorithms for tensors.
///
/// Provides entropy, divergence, and mutual information measures.
pub trait InformationTheoryAlgorithms<R: Runtime<DType = DType>>: TensorOps<R> {
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

    /// Cross-entropy between two probability distributions.
    ///
    /// H(p, q) = -Σ p(x) log(q(x))
    ///
    /// Measures the average number of bits/nats needed to identify events from
    /// distribution p when using a code optimized for distribution q.
    ///
    /// # Arguments
    ///
    /// * `pk` - True distribution (1-D tensor, must sum to 1)
    /// * `qk` - Predicted distribution (1-D tensor, must sum to 1, same length as pk)
    /// * `base` - Logarithm base (e.g., 2.0 for bits). If None, uses natural log.
    fn cross_entropy(&self, pk: &Tensor<R>, qk: &Tensor<R>, base: Option<f64>)
    -> Result<Tensor<R>>;

    /// Negative log-likelihood loss.
    ///
    /// `NLL = -mean(log(p[target]))`
    ///
    /// For each sample, selects the predicted probability at the target index,
    /// takes its logarithm, negates, and averages over all samples.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log-probabilities [N, C] (e.g., output of log_softmax)
    /// * `targets` - Target class indices `[N]` (integer tensor, values in `[0, C)`)
    fn nll_loss(&self, log_probs: &Tensor<R>, targets: &Tensor<R>) -> Result<Tensor<R>>;
}
