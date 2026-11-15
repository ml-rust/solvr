//! CPU batched bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bisect_batched_impl;
use crate::optimize::roots::{
    BatchedBisectionAlgorithms, BatchedBisectionOptions, BatchedBracketResult,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BatchedBisectionAlgorithms<CpuRuntime> for CpuClient {
    fn bisect_batched<F>(
        &self,
        f: F,
        lower: &Tensor<CpuRuntime>,
        upper: &Tensor<CpuRuntime>,
        options: &BatchedBisectionOptions,
    ) -> OptimizeResult<BatchedBracketResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> numr::error::Result<Tensor<CpuRuntime>>,
    {
        bisect_batched_impl(self, f, lower, upper, options)
    }
}

#[cfg(test)]
#[path = "bisection_tests.rs"]
mod tests;
