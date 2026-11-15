//! CPU certified upper-bracket delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bracket_upper_certified_impl;
use crate::optimize::roots::{
    CertifiedRootQuery, CertifiedRootValues, CertifiedUpperBracketAlgorithms,
    CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CertifiedUpperBracketAlgorithms<CpuRuntime> for CpuClient {
    fn bracket_upper_certified<F>(
        &self,
        f: F,
        lower: &Tensor<CpuRuntime>,
        initial_upper: &Tensor<CpuRuntime>,
        upper_limit: &Tensor<CpuRuntime>,
        options: &CertifiedUpperBracketOptions,
    ) -> OptimizeResult<CertifiedUpperBracketResult<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
    {
        bracket_upper_certified_impl(self, f, lower, initial_upper, upper_limit, options)
    }
}
