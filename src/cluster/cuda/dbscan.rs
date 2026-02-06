//! CUDA implementation of DBSCAN clustering.

use crate::cluster::impl_generic::dbscan_impl;
use crate::cluster::traits::dbscan::{DbscanAlgorithms, DbscanOptions, DbscanResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl DbscanAlgorithms<CudaRuntime> for CudaClient {
    fn dbscan(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &DbscanOptions,
    ) -> Result<DbscanResult<CudaRuntime>> {
        dbscan_impl(self, data, options)
    }
}
