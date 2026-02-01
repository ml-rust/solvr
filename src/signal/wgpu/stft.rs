//! WebGPU implementation of STFT and inverse STFT algorithms.

use crate::signal::impl_generic::{istft_impl, stft_impl};
use crate::signal::traits::stft::StftAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl StftAlgorithms<WgpuRuntime> for WgpuClient {
    fn stft(
        &self,
        signal: &Tensor<WgpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<WgpuRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
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
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_stft_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<WgpuRuntime>::from_slice(&signal, &[256], &device);

        let result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (256 + 64 - 64) / 16 + 1;

        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
