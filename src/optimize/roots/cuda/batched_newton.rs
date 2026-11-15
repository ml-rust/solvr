//! Cuda batched Newton correction delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::newton_batched_impl;
use crate::optimize::roots::{BatchedNewtonAlgorithms, BatchedNewtonOptions, BatchedNewtonResult};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BatchedNewtonAlgorithms<CudaRuntime> for CudaClient {
    fn newton_batched<F>(
        &self,
        correction_callback: F,
        initial: &Tensor<CudaRuntime>,
        options: &BatchedNewtonOptions,
    ) -> OptimizeResult<BatchedNewtonResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> numr::error::Result<Tensor<CudaRuntime>>,
    {
        newton_batched_impl(self, correction_callback, initial, options)
    }
}

#[cfg(test)]
#[path = "batched_newton_tests.rs"]
mod tests;
