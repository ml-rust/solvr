//! CUDA mapped certified bisection delegation.

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
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CertifiedMappedBisectionAlgorithms<CudaRuntime> for CudaClient {
    fn bisect_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        lower: &Tensor<CudaRuntime>,
        upper: &Tensor<CudaRuntime>,
        derivative_lower: &Tensor<CudaRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CudaRuntime>>,
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
        continuation: CertifiedRootContinuation<CudaRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CudaRuntime>>,
    {
        resume_certified_mapped_impl(self, residual, mapping, continuation, tolerance, allowance)
    }
    fn resume_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<CudaRuntime>,
        owners: &Tensor<CudaRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CudaRuntime>>,
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
        continuation: CertifiedRootContinuation<CudaRuntime>,
        owners: &Tensor<CudaRuntime>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<CudaRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CudaRuntime>>,
        M: Fn(
            &CertifiedRootDomainQuery<CudaRuntime>,
        ) -> numr::error::Result<CertifiedRootMappedValues<CudaRuntime>>,
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
