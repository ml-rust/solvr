//! WGPU certified upper-bracket delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bracket_upper_certified_impl;
use crate::optimize::roots::{
    CertifiedRootQuery, CertifiedRootValues, CertifiedUpperBracketAlgorithms,
    CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CertifiedUpperBracketAlgorithms<WgpuRuntime> for WgpuClient {
    fn bracket_upper_certified<F>(
        &self,
        f: F,
        lower: &Tensor<WgpuRuntime>,
        initial_upper: &Tensor<WgpuRuntime>,
        upper_limit: &Tensor<WgpuRuntime>,
        options: &CertifiedUpperBracketOptions,
    ) -> OptimizeResult<CertifiedUpperBracketResult<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
    {
        bracket_upper_certified_impl(self, f, lower, initial_upper, upper_limit, options)
    }
}
