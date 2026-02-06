//! CUDA implementation of HDBSCAN clustering.

use crate::cluster::impl_generic::hdbscan_impl;
use crate::cluster::traits::hdbscan::{HdbscanAlgorithms, HdbscanOptions, HdbscanResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl HdbscanAlgorithms<CudaRuntime> for CudaClient {
    fn hdbscan(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &HdbscanOptions,
    ) -> Result<HdbscanResult<CudaRuntime>> {
        hdbscan_impl(self, data, options)
    }
}
