//! WebGPU delegation for certified fixed Simpson integration.

use crate::integrate::error::IntegrateResult;
use crate::integrate::impl_generic::quadrature::certified_simpson_impl;
use crate::integrate::traits::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonPanel,
    CertifiedSimpsonQuery, CertifiedSimpsonResult, CertifiedSimpsonValues,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CertifiedSimpsonAlgorithms<WgpuRuntime> for WgpuClient {
    fn certified_simpson<F, D>(
        &self,
        left: &Tensor<WgpuRuntime>,
        right: &Tensor<WgpuRuntime>,
        options: CertifiedSimpsonOptions,
        values: F,
        fourth_bound: D,
    ) -> IntegrateResult<CertifiedSimpsonResult<WgpuRuntime>>
    where
        F: FnMut(
            &CertifiedSimpsonQuery<WgpuRuntime>,
        ) -> IntegrateResult<CertifiedSimpsonValues<WgpuRuntime>>,
        D: FnMut(&CertifiedSimpsonPanel<WgpuRuntime>) -> IntegrateResult<Tensor<WgpuRuntime>>,
    {
        certified_simpson_impl(self, left, right, options, values, fourth_bound)
    }
}
