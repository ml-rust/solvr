//! Cpu batched Newton correction delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::newton_batched_impl;
use crate::optimize::roots::{BatchedNewtonAlgorithms, BatchedNewtonOptions, BatchedNewtonResult};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BatchedNewtonAlgorithms<CpuRuntime> for CpuClient {
    fn newton_batched<F>(
        &self,
        correction_callback: F,
        initial: &Tensor<CpuRuntime>,
        options: &BatchedNewtonOptions,
    ) -> OptimizeResult<BatchedNewtonResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> numr::error::Result<Tensor<CpuRuntime>>,
    {
        newton_batched_impl(self, correction_callback, initial, options)
    }
}

#[cfg(test)]
#[path = "batched_newton_tests.rs"]
mod tests;
