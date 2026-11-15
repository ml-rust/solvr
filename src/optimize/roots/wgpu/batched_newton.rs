//! Wgpu batched Newton correction delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::newton_batched_impl;
use crate::optimize::roots::{BatchedNewtonAlgorithms, BatchedNewtonOptions, BatchedNewtonResult};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BatchedNewtonAlgorithms<WgpuRuntime> for WgpuClient {
    fn newton_batched<F>(
        &self,
        correction_callback: F,
        initial: &Tensor<WgpuRuntime>,
        options: &BatchedNewtonOptions,
    ) -> OptimizeResult<BatchedNewtonResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> numr::error::Result<Tensor<WgpuRuntime>>,
    {
        if initial.dtype() == crate::DType::F64 {
            return Err(crate::optimize::OptimizeError::InvalidInput {
                context: "newton_batched: WGPU requires F32 input, F64 is unsupported".into(),
            });
        }
        newton_batched_impl(self, correction_callback, initial, options)
    }
}

#[cfg(test)]
#[path = "batched_newton_tests.rs"]
mod tests;
