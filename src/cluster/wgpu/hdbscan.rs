//! WebGPU implementation of HDBSCAN clustering.

use crate::cluster::impl_generic::hdbscan_impl;
use crate::cluster::traits::hdbscan::{HdbscanAlgorithms, HdbscanOptions, HdbscanResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl HdbscanAlgorithms<WgpuRuntime> for WgpuClient {
    fn hdbscan(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &HdbscanOptions,
    ) -> Result<HdbscanResult<WgpuRuntime>> {
        hdbscan_impl(self, data, options)
    }
}
