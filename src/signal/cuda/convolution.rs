//! CUDA implementation of convolution and cross-correlation algorithms.

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl,
};
use crate::signal::traits::convolution::{ConvMode, ConvolutionAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ConvolutionAlgorithms<CudaRuntime> for CudaClient {
    fn convolve(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        // Skip if no CUDA device available
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_convolve_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal =
            Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[7]);
    }
}
