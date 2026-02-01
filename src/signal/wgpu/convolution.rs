//! WebGPU implementation of convolution and cross-correlation algorithms.

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl,
};
use crate::signal::traits::convolution::{ConvMode, ConvolutionAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ConvolutionAlgorithms<WgpuRuntime> for WgpuClient {
    fn convolve(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_convolve_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        // WebGPU only supports F32
        let signal =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[7]);
    }
}
