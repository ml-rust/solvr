//! WebGPU implementation of Mean Shift clustering.

use crate::cluster::impl_generic::mean_shift_impl;
use crate::cluster::traits::mean_shift::{MeanShiftAlgorithms, MeanShiftOptions, MeanShiftResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MeanShiftAlgorithms<WgpuRuntime> for WgpuClient {
    fn mean_shift(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &MeanShiftOptions,
    ) -> Result<MeanShiftResult<WgpuRuntime>> {
        mean_shift_impl(self, data, options)
    }
}
