//! CPU certified bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bisect_certified_impl;
use crate::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootQuery,
    CertifiedRootResult, CertifiedRootValues,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CertifiedBisectionAlgorithms<CpuRuntime> for CpuClient {
    fn bisect_certified<F>(
        &self,
        f: F,
        lower: &Tensor<CpuRuntime>,
        upper: &Tensor<CpuRuntime>,
        derivative_lower: &Tensor<CpuRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedRootResult<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
    {
        bisect_certified_impl(self, f, lower, upper, derivative_lower, options)
    }
}
