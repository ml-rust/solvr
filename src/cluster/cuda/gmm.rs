//! CUDA implementation of Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    gmm_fit_impl, gmm_predict_impl, gmm_predict_proba_impl, gmm_score_impl,
};
use crate::cluster::traits::gmm::{GmmAlgorithms, GmmModel, GmmOptions};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl GmmAlgorithms<CudaRuntime> for CudaClient {
    fn gmm_fit(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &GmmOptions,
    ) -> Result<GmmModel<CudaRuntime>> {
        gmm_fit_impl(self, data, options)
    }

    fn gmm_predict(
        &self,
        model: &GmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gmm_predict_impl(self, model, data)
    }

    fn gmm_predict_proba(
        &self,
        model: &GmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gmm_predict_proba_impl(self, model, data)
    }

    fn gmm_score(
        &self,
        model: &GmmModel<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gmm_score_impl(self, model, data)
    }
}
