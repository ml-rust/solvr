//! Bayesian Gaussian Mixture Model trait.
//!
//! Variational inference with Dirichlet process priors, allowing automatic
//! determination of the number of active components.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::gmm::{CovarianceType, GmmInit};

/// Weight concentration prior type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightConcentrationPrior {
    /// Dirichlet process (stick-breaking).
    #[default]
    DirichletProcess,
    /// Symmetric Dirichlet distribution.
    DirichletDistribution,
}

/// Options for Bayesian Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct BayesianGmmOptions {
    /// Maximum number of mixture components.
    pub n_components: usize,
    /// Covariance type.
    pub covariance_type: CovarianceType,
    /// Maximum variational EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on ELBO change.
    pub tol: f64,
    /// Number of random restarts.
    pub n_init: usize,
    /// Initialization method.
    pub init: GmmInit,
    /// Regularization added to covariance diagonal.
    pub reg_covar: f64,
    /// Weight concentration prior type.
    pub weight_concentration_prior_type: WeightConcentrationPrior,
    /// Concentration parameter for the weight prior.
    /// For Dirichlet process: higher = more uniform weights.
    /// None = 1/n_components.
    pub weight_concentration_prior: Option<f64>,
    /// Prior on mean precision (beta_0). None = 1.0.
    pub mean_precision_prior: Option<f64>,
    /// Degrees of freedom for Wishart prior. None = d (dimensionality).
    pub degrees_of_freedom_prior: Option<f64>,
}

impl Default for BayesianGmmOptions {
    fn default() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            n_init: 1,
            init: GmmInit::KMeans,
            reg_covar: 1e-6,
            weight_concentration_prior_type: WeightConcentrationPrior::DirichletProcess,
            weight_concentration_prior: None,
            mean_precision_prior: None,
            degrees_of_freedom_prior: None,
        }
    }
}

/// Fitted Bayesian Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct BayesianGmmModel<R: Runtime> {
    /// Effective mixture weights [k] (some may be near zero).
    pub weights: Tensor<R>,
    /// Component means [k, d].
    pub means: Tensor<R>,
    /// Covariances (shape depends on covariance_type).
    pub covariances: Tensor<R>,
    /// Precomputed precision information.
    pub precisions_cholesky: Tensor<R>,
    /// Weight concentration parameters (posterior).
    pub weight_concentration: Tensor<R>,
    /// Mean precision parameters (posterior) [k].
    pub mean_precision: Tensor<R>,
    /// Degrees of freedom (posterior) [k].
    pub degrees_of_freedom: Tensor<R>,
    /// Whether variational EM converged.
    pub converged: bool,
    /// Number of iterations run.
    pub n_iter: usize,
    /// Final evidence lower bound (ELBO).
    pub lower_bound: f64,
}

/// Bayesian Gaussian Mixture Model algorithms.
pub trait BayesianGmmAlgorithms<R: Runtime> {
    /// Fit Bayesian GMM to data [n, d].
    fn bayesian_gmm_fit(
        &self,
        data: &Tensor<R>,
        options: &BayesianGmmOptions,
    ) -> Result<BayesianGmmModel<R>>;

    /// Predict most likely component for each point.
    fn bayesian_gmm_predict(
        &self,
        model: &BayesianGmmModel<R>,
        data: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Predict component probabilities [n, k].
    fn bayesian_gmm_predict_proba(
        &self,
        model: &BayesianGmmModel<R>,
        data: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Compute per-sample log-likelihood under the model.
    fn bayesian_gmm_score(
        &self,
        model: &BayesianGmmModel<R>,
        data: &Tensor<R>,
    ) -> Result<Tensor<R>>;
}
