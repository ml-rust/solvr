//! WGPU mapped certified bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::{
    bisect_certified_mapped_impl, refresh_certified_mapped_selected_impl,
    resume_certified_mapped_impl, resume_certified_mapped_selected_impl,
};
use crate::optimize::roots::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedMappedRootProgress,
    CertifiedRootAdvanceAllowance, CertifiedRootContinuation, CertifiedRootDomainQuery,
    CertifiedRootMappedValues, CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CertifiedMappedBisectionAlgorithms<WgpuRuntime> for WgpuClient {
    fn bisect_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        lower: &Tensor<WgpuRuntime>,
        upper: &Tensor<WgpuRuntime>,
        derivative_lower: &Tensor<WgpuRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedMappedRootProgress<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<WgpuRuntime>>,
    {
        bisect_certified_mapped_impl(
            self,
            residual,
            mapping,
            lower,
            upper,
            derivative_lower,
            options,
        )
    }

    fn resume_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<WgpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<WgpuRuntime>>,
    {
        resume_certified_mapped_impl(self, residual, mapping, continuation, tolerance, allowance)
    }
    fn resume_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<WgpuRuntime>,
        owners: &Tensor<WgpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<WgpuRuntime>>,
    {
        resume_certified_mapped_selected_impl(
            self,
            residual,
            mapping,
            continuation,
            owners,
            tolerance,
            allowance,
        )
    }
    fn refresh_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<WgpuRuntime>,
        owners: &Tensor<WgpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<WgpuRuntime>>,
    {
        refresh_certified_mapped_selected_impl(
            self,
            residual,
            mapping,
            continuation,
            owners,
            tolerance,
            allowance,
        )
    }
}
