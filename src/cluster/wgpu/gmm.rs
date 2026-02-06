//! WebGPU implementation of Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    gmm_fit_impl, gmm_predict_impl, gmm_predict_proba_impl, gmm_score_impl,
};
use crate::cluster::traits::gmm::{GmmAlgorithms, GmmModel, GmmOptions};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl GmmAlgorithms<WgpuRuntime> for WgpuClient {
    fn gmm_fit(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &GmmOptions,
    ) -> Result<GmmModel<WgpuRuntime>> {
        gmm_fit_impl(self, data, options)
    }

    fn gmm_predict(
        &self,
        model: &GmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        gmm_predict_impl(self, model, data)
    }

    fn gmm_predict_proba(
        &self,
        model: &GmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        gmm_predict_proba_impl(self, model, data)
    }

    fn gmm_score(
        &self,
        model: &GmmModel<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        gmm_score_impl(self, model, data)
    }
}
