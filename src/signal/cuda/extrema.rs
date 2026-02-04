//! CUDA implementation of local extrema detection algorithms.
//!
//! This algorithm is CPU-only due to its sequential comparison patterns.
//! GPU implementations are not efficient for variable-order neighborhood comparisons.

use crate::signal::traits::extrema::{ExtremaAlgorithms, ExtremaResult, ExtremumMode};
use numr::error::{Error, Result};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ExtremaAlgorithms<CudaRuntime> for CudaClient {
    fn argrelmin(
        &self,
        _x: &Tensor<CudaRuntime>,
        _order: usize,
        _mode: ExtremumMode,
    ) -> Result<ExtremaResult<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "argrelmin",
            reason: "Local extrema detection is CPU-only due to sequential comparison patterns. Transfer data to CPU first.".to_string(),
        })
    }

    fn argrelmax(
        &self,
        _x: &Tensor<CudaRuntime>,
        _order: usize,
        _mode: ExtremumMode,
    ) -> Result<ExtremaResult<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "argrelmax",
            reason: "Local extrema detection is CPU-only due to sequential comparison patterns. Transfer data to CPU first.".to_string(),
        })
    }
}
