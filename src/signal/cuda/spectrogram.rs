//! CUDA implementation of spectrogram algorithm.

use crate::signal::impl_generic::spectrogram_impl;
use crate::signal::traits::spectrogram::SpectrogramAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SpectrogramAlgorithms<CudaRuntime> for CudaClient {
    fn spectrogram(
        &self,
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        power: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
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
    fn test_spectrogram_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
        let signal_tensor = Tensor::<CudaRuntime>::from_slice(&signal, &[512], &device);

        let result = client
            .spectrogram(&signal_tensor, 64, Some(32), None, 2.0)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (512 + 64 - 64) / 32 + 1;
        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
