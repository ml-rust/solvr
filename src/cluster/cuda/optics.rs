//! CUDA implementation of OPTICS clustering.

use crate::cluster::impl_generic::optics_impl;
use crate::cluster::traits::optics::{OpticsAlgorithms, OpticsOptions, OpticsResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl OpticsAlgorithms<CudaRuntime> for CudaClient {
    fn optics(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &OpticsOptions,
    ) -> Result<OpticsResult<CudaRuntime>> {
        optics_impl(self, data, options)
    }
}
