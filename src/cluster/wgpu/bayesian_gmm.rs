//! WebGPU implementation of Bayesian Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    bayesian_gmm_fit_impl, bayesian_gmm_predict_impl, bayesian_gmm_predict_proba_impl,
    bayesian_gmm_score_impl,
};
use crate::cluster::traits::bayesian_gmm::{
    BayesianGmmAlgorithms, BayesianGmmModel, BayesianGmmOptions,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BayesianGmmAlgorithms<WgpuRuntime> for WgpuClient {
    fn bayesian_gmm_fit(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &BayesianGmmOptions,
    ) -> Result<BayesianGmmModel<WgpuRuntime>> {
        bayesian_gmm_fit_impl(self, data, options)
    }

    fn bayesian_gmm_predict(
        &self,
        model: &BayesianGmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        bayesian_gmm_predict_impl(self, model, data)
    }

    fn bayesian_gmm_predict_proba(
        &self,
        model: &BayesianGmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        bayesian_gmm_predict_proba_impl(self, model, data)
    }

    fn bayesian_gmm_score(
        &self,
        model: &BayesianGmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        bayesian_gmm_score_impl(self, model, data)
    }
}
