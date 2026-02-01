//! CUDA implementation of STFT and inverse STFT algorithms.

use crate::signal::impl_generic::{istft_impl, stft_impl};
use crate::signal::traits::stft::StftAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl StftAlgorithms<CudaRuntime> for CudaClient {
    fn stft(
        &self,
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CudaRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        istft_impl(
            self,
            stft_matrix,
            hop_length,
            window,
            center,
            length,
            normalized,
        )
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
    fn test_stft_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<CudaRuntime>::from_slice(&signal, &[256], &device);

        let result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (256 + 64 - 64) / 16 + 1;

        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
