//! Linear diagnostic receipts for the cpu runtime through shared tensor operations.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::linear_diagnostics::interp1d_linear_with_receipts_impl;
use crate::interpolate::traits::linear_diagnostics::{
    LinearInterp1dDiagnostics, LinearInterp1dReceipt,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl LinearInterp1dDiagnostics<CpuRuntime> for CpuClient {
    fn interp1d_linear_with_receipts(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<LinearInterp1dReceipt<CpuRuntime>> {
        interp1d_linear_with_receipts_impl(self, x, y, query)
    }
}
