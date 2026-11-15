//! CUDA certified upper-bracket delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bracket_upper_certified_impl;
use crate::optimize::roots::{
    CertifiedRootQuery, CertifiedRootValues, CertifiedUpperBracketAlgorithms,
    CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CertifiedUpperBracketAlgorithms<CudaRuntime> for CudaClient {
    fn bracket_upper_certified<F>(
        &self,
        f: F,
        lower: &Tensor<CudaRuntime>,
        initial_upper: &Tensor<CudaRuntime>,
        upper_limit: &Tensor<CudaRuntime>,
        options: &CertifiedUpperBracketOptions,
    ) -> OptimizeResult<CertifiedUpperBracketResult<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
    {
        bracket_upper_certified_impl(self, f, lower, initial_upper, upper_limit, options)
    }
}
