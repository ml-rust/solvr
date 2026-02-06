//! WebGPU implementation of DBSCAN clustering.

use crate::cluster::impl_generic::dbscan_impl;
use crate::cluster::traits::dbscan::{DbscanAlgorithms, DbscanOptions, DbscanResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl DbscanAlgorithms<WgpuRuntime> for WgpuClient {
    fn dbscan(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &DbscanOptions,
    ) -> Result<DbscanResult<WgpuRuntime>> {
        dbscan_impl(self, data, options)
    }
}
