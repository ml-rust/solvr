//! WebGPU implementation of Affinity Propagation clustering.

use crate::cluster::impl_generic::affinity_propagation_impl;
use crate::cluster::traits::affinity_propagation::{
    AffinityPropagationAlgorithms, AffinityPropagationOptions, AffinityPropagationResult,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl AffinityPropagationAlgorithms<WgpuRuntime> for WgpuClient {
    fn affinity_propagation(
        &self,
        similarities: &Tensor<WgpuRuntime>,
        options: &AffinityPropagationOptions,
    ) -> Result<AffinityPropagationResult<WgpuRuntime>> {
        affinity_propagation_impl(self, similarities, options)
    }
}
