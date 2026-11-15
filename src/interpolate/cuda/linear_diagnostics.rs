//! Linear diagnostic receipts for the cuda runtime through shared tensor operations.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::linear_diagnostics::interp1d_linear_with_receipts_impl;
use crate::interpolate::traits::linear_diagnostics::{
    LinearInterp1dDiagnostics, LinearInterp1dReceipt,
};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl LinearInterp1dDiagnostics<CudaRuntime> for CudaClient {
    fn interp1d_linear_with_receipts(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<LinearInterp1dReceipt<CudaRuntime>> {
        interp1d_linear_with_receipts_impl(self, x, y, query)
    }
}
