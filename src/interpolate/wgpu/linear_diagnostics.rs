//! Linear diagnostic receipts for the wgpu runtime through shared tensor operations.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::linear_diagnostics::interp1d_linear_with_receipts_impl;
use crate::interpolate::traits::linear_diagnostics::{
    LinearInterp1dDiagnostics, LinearInterp1dReceipt,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl LinearInterp1dDiagnostics<WgpuRuntime> for WgpuClient {
    fn interp1d_linear_with_receipts(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<LinearInterp1dReceipt<WgpuRuntime>> {
        interp1d_linear_with_receipts_impl(self, x, y, query)
    }
}
