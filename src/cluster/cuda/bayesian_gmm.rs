//! CUDA implementation of Bayesian Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    bayesian_gmm_fit_impl, bayesian_gmm_predict_impl, bayesian_gmm_predict_proba_impl,
    bayesian_gmm_score_impl,
};
use crate::cluster::traits::bayesian_gmm::{
    BayesianGmmAlgorithms, BayesianGmmModel, BayesianGmmOptions,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BayesianGmmAlgorithms<CudaRuntime> for CudaClient {
    fn bayesian_gmm_fit(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &BayesianGmmOptions,
    ) -> Result<BayesianGmmModel<CudaRuntime>> {
        bayesian_gmm_fit_impl(self, data, options)
    }

    fn bayesian_gmm_predict(
        &self,
        model: &BayesianGmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        bayesian_gmm_predict_impl(self, model, data)
    }

    fn bayesian_gmm_predict_proba(
        &self,
        model: &BayesianGmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        bayesian_gmm_predict_proba_impl(self, model, data)
    }

    fn bayesian_gmm_score(
        &self,
        model: &BayesianGmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        bayesian_gmm_score_impl(self, model, data)
    }
}
