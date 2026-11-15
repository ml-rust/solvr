//! CPU mapped certified bisection delegation.

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
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CertifiedMappedBisectionAlgorithms<CpuRuntime> for CpuClient {
    fn bisect_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        lower: &Tensor<CpuRuntime>,
        upper: &Tensor<CpuRuntime>,
        derivative_lower: &Tensor<CpuRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CpuRuntime>>,
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
        continuation: CertifiedRootContinuation<CpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CpuRuntime>>,
    {
        resume_certified_mapped_impl(self, residual, mapping, continuation, tolerance, allowance)
    }
    fn resume_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<CpuRuntime>,
        owners: &Tensor<CpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CpuRuntime>>,
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
        continuation: CertifiedRootContinuation<CpuRuntime>,
        owners: &Tensor<CpuRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CpuRuntime>>,
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
