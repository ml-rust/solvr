//! Gaussian Mixture Model trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Covariance parameterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceType {
    /// Each component has its own full covariance [k, d, d].
    #[default]
    Full,
    /// All components share one covariance [d, d].
    Tied,
    /// Diagonal covariance [k, d].
    Diagonal,
    /// Scalar variance per component [k].
    Spherical,
}

/// Initialization method for GMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GmmInit {
    /// Initialize via K-Means.
    #[default]
    KMeans,
    /// Random initialization.
    Random,
}

/// Options for Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GmmOptions {
    /// Number of mixture components.
    pub n_components: usize,
    /// Covariance type.
    pub covariance_type: CovarianceType,
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood.
    pub tol: f64,
    /// Number of random restarts.
    pub n_init: usize,
    /// Initialization method.
    pub init: GmmInit,
    /// Regularization added to covariance diagonal.
    pub reg_covar: f64,
}

impl Default for GmmOptions {
    fn default() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            n_init: 1,
            init: GmmInit::KMeans,
            reg_covar: 1e-6,
        }
    }
}

/// Fitted Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GmmModel<R: Runtime> {
    /// Mixture weights [k] (sum = 1).
    pub weights: Tensor<R>,
    /// Component means [k, d].
    pub means: Tensor<R>,
    /// Covariances (shape depends on covariance_type).
    pub covariances: Tensor<R>,
    /// Precomputed Cholesky of precision matrices.
    pub precisions_cholesky: Tensor<R>,
    /// Whether EM converged.
    pub converged: bool,
    /// Number of iterations run.
    pub n_iter: usize,
    /// Final log-likelihood lower bound.
    pub lower_bound: f64,
}

/// Gaussian Mixture Model algorithms.
pub trait GmmAlgorithms<R: Runtime> {
    /// Fit GMM to data [n, d].
    fn gmm_fit(&self, data: &Tensor<R>, options: &GmmOptions) -> Result<GmmModel<R>>;

    /// Predict most likely component for each point.
    fn gmm_predict(&self, model: &GmmModel<R>, data: &Tensor<R>) -> Result<Tensor<R>>;

    /// Predict component probabilities [n, k].
    fn gmm_predict_proba(&self, model: &GmmModel<R>, data: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute per-sample log-likelihood.
    fn gmm_score(&self, model: &GmmModel<R>, data: &Tensor<R>) -> Result<Tensor<R>>;
}
