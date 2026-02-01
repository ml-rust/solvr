//! CPU implementation of convolution and cross-correlation algorithms.

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl,
};
use crate::signal::traits::convolution::{ConvMode, ConvolutionAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ConvolutionAlgorithms<CpuRuntime> for CpuClient {
    fn convolve(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_convolve_full() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, -1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[7]);
        let data: Vec<f64> = result.to_vec();
        // Expected: [1, 2, 2, 2, 2, -4, -5]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve_same() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Same).unwrap();

        assert_eq!(result.shape(), &[5]);
    }

    #[test]
    fn test_convolve_valid() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Valid).unwrap();

        assert_eq!(result.shape(), &[3]);
        let data: Vec<f64> = result.to_vec();
        // Sum of windows: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve2d() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            &device,
        );
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], &device);

        let result = client.convolve2d(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
    }

    #[test]
    fn test_correlate() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device);

        let result = client.correlate(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[6]);
    }

    #[test]
    fn test_convolve_f32() {
        let (client, device) = setup();

        let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.dtype(), numr::dtype::DType::F32);
        assert_eq!(result.shape(), &[6]);
    }
}
