//! WGPU certified bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bisect_certified_impl;
use crate::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootQuery,
    CertifiedRootResult, CertifiedRootValues,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CertifiedBisectionAlgorithms<WgpuRuntime> for WgpuClient {
    fn bisect_certified<F>(
        &self,
        f: F,
        lower: &Tensor<WgpuRuntime>,
        upper: &Tensor<WgpuRuntime>,
        derivative_lower: &Tensor<WgpuRuntime>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedRootResult<WgpuRuntime>>
    where
        F: Fn(
            &CertifiedRootQuery<WgpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<WgpuRuntime>>,
    {
        bisect_certified_impl(self, f, lower, upper, derivative_lower, options)
    }
}
