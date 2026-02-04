//! CUDA implementation of Wiener filter algorithms.
//!
//! Wiener filtering is CPU-only because it requires computing local statistics
//! over sliding windows with sequential access patterns.

use crate::signal::traits::wiener::WienerAlgorithms;
use numr::error::{Error, Result};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl WienerAlgorithms<CudaRuntime> for CudaClient {
    fn wiener(
        &self,
        _x: &Tensor<CudaRuntime>,
        _kernel_size: Option<usize>,
        _noise: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "wiener",
            reason: "Wiener filtering is CPU-only due to local statistics computation. Transfer data to CPU first.".to_string(),
        })
    }

    fn wiener2d(
        &self,
        _x: &Tensor<CudaRuntime>,
        _kernel_size: Option<(usize, usize)>,
        _noise: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            op: "wiener2d",
            reason: "Wiener filtering is CPU-only due to local statistics computation. Transfer data to CPU first.".to_string(),
        })
    }
}
