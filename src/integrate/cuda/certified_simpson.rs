//! CUDA delegation for certified fixed Simpson integration.

use crate::integrate::error::IntegrateResult;
use crate::integrate::impl_generic::quadrature::certified_simpson_impl;
use crate::integrate::traits::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonPanel,
    CertifiedSimpsonQuery, CertifiedSimpsonResult, CertifiedSimpsonValues,
};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CertifiedSimpsonAlgorithms<CudaRuntime> for CudaClient {
    fn certified_simpson<F, D>(
        &self,
        left: &Tensor<CudaRuntime>,
        right: &Tensor<CudaRuntime>,
        options: CertifiedSimpsonOptions,
        values: F,
        fourth_bound: D,
    ) -> IntegrateResult<CertifiedSimpsonResult<CudaRuntime>>
    where
        F: FnMut(
            &CertifiedSimpsonQuery<CudaRuntime>,
        ) -> IntegrateResult<CertifiedSimpsonValues<CudaRuntime>>,
        D: FnMut(&CertifiedSimpsonPanel<CudaRuntime>) -> IntegrateResult<Tensor<CudaRuntime>>,
    {
        certified_simpson_impl(self, left, right, options, values, fourth_bound)
    }
}
