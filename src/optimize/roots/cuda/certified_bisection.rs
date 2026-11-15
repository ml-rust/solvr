//! CUDA certified bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bisect_certified_impl;
use crate::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootQuery,
    CertifiedRootResult, CertifiedRootValues,
};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CertifiedBisectionAlgorithms<CudaRuntime> for CudaClient {
    fn bisect_certified<F>(
        &self,
        f: F,
        lower: &Tensor<CudaRuntime>,
        upper: &Tensor<CudaRuntime>,
        derivative_lower: &Tensor<CudaRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedRootResult<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
    {
        bisect_certified_impl(self, f, lower, upper, derivative_lower, options)
    }
}
