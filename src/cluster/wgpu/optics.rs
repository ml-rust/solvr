//! WebGPU implementation of OPTICS clustering.

use crate::cluster::impl_generic::optics_impl;
use crate::cluster::traits::optics::{OpticsAlgorithms, OpticsOptions, OpticsResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl OpticsAlgorithms<WgpuRuntime> for WgpuClient {
    fn optics(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &OpticsOptions,
    ) -> Result<OpticsResult<WgpuRuntime>> {
        optics_impl(self, data, options)
    }
}
