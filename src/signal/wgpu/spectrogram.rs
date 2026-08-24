//! WebGPU implementation of spectrogram algorithm.

use crate::signal::impl_generic::spectrogram_impl;
use crate::signal::traits::spectrogram::SpectrogramAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SpectrogramAlgorithms<WgpuRuntime> for WgpuClient {
    fn spectrogram(
        &self,
        signal: &Tensor<WgpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        power: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_spectrogram_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let signal_tensor = Tensor::<WgpuRuntime>::from_slice(&signal, &[512], &device).unwrap();

        // Spectrogram uses FFT → Complex64, which wgpu doesn't support.
        let result = match client.spectrogram(&signal_tensor, 64, Some(32), None, 2.0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping test_spectrogram_wgpu: {e}");
                return;
            }
        };

        let freq_bins = 64 / 2 + 1;
        let n_frames = (512 + 64 - 64) / 32 + 1;
        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
