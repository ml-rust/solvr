//! CUDA implementation of Affinity Propagation clustering.

use crate::cluster::impl_generic::affinity_propagation_impl;
use crate::cluster::traits::affinity_propagation::{
    AffinityPropagationAlgorithms, AffinityPropagationOptions, AffinityPropagationResult,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl AffinityPropagationAlgorithms<CudaRuntime> for CudaClient {
    fn affinity_propagation(
        &self,
        similarities: &Tensor<CudaRuntime>,
        options: &AffinityPropagationOptions,
    ) -> Result<AffinityPropagationResult<CudaRuntime>> {
        affinity_propagation_impl(self, similarities, options)
    }
}
