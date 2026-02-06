//! CUDA implementation of Mean Shift clustering.

use crate::cluster::impl_generic::mean_shift_impl;
use crate::cluster::traits::mean_shift::{MeanShiftAlgorithms, MeanShiftOptions, MeanShiftResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MeanShiftAlgorithms<CudaRuntime> for CudaClient {
    fn mean_shift(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &MeanShiftOptions,
    ) -> Result<MeanShiftResult<CudaRuntime>> {
        mean_shift_impl(self, data, options)
    }
}
