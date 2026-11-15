//! CPU delegation for certified fixed Simpson integration.

use crate::integrate::error::IntegrateResult;
use crate::integrate::impl_generic::quadrature::certified_simpson_impl;
use crate::integrate::traits::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonPanel,
    CertifiedSimpsonQuery, CertifiedSimpsonResult, CertifiedSimpsonValues,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CertifiedSimpsonAlgorithms<CpuRuntime> for CpuClient {
    fn certified_simpson<F, D>(
        &self,
        left: &Tensor<CpuRuntime>,
        right: &Tensor<CpuRuntime>,
        options: CertifiedSimpsonOptions,
        values: F,
        fourth_bound: D,
    ) -> IntegrateResult<CertifiedSimpsonResult<CpuRuntime>>
    where
        F: FnMut(
            &CertifiedSimpsonQuery<CpuRuntime>,
        ) -> IntegrateResult<CertifiedSimpsonValues<CpuRuntime>>,
        D: FnMut(&CertifiedSimpsonPanel<CpuRuntime>) -> IntegrateResult<Tensor<CpuRuntime>>,
    {
        certified_simpson_impl(self, left, right, options, values, fourth_bound)
    }
}
