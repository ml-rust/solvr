//! WebGPU implementation of local extrema detection algorithms.
//!
//! This algorithm is CPU-only due to its sequential comparison patterns.
//! GPU implementations are not efficient for variable-order neighborhood comparisons.

use crate::signal::traits::extrema::{ExtremaAlgorithms, ExtremaResult, ExtremumMode};
use numr::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ExtremaAlgorithms<WgpuRuntime> for WgpuClient {
    fn argrelmin(
        &self,
        _x: &Tensor<WgpuRuntime>,
        _order: usize,
        _mode: ExtremumMode,
    ) -> Result<ExtremaResult<WgpuRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "argrelmin",
            reason: "Local extrema detection is CPU-only due to sequential comparison patterns. Transfer data to CPU first.".to_string(),
        })
    }

    fn argrelmax(
        &self,
        _x: &Tensor<WgpuRuntime>,
        _order: usize,
        _mode: ExtremumMode,
    ) -> Result<ExtremaResult<WgpuRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "argrelmax",
            reason: "Local extrema detection is CPU-only due to sequential comparison patterns. Transfer data to CPU first.".to_string(),
        })
    }
}
