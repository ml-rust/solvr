//! CPU implementation of spectrogram algorithm.

use crate::signal::impl_generic::spectrogram_impl;
use crate::signal::traits::spectrogram::SpectrogramAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SpectrogramAlgorithms<CpuRuntime> for CpuClient {
    fn spectrogram(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        power: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
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
    fn test_spectrogram() {
        let (client, device) = setup();

        let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
        let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[512], &device);

        let result = client
            .spectrogram(&signal_tensor, 64, Some(32), None, 2.0)
            .unwrap();

        // Power spectrogram should be real-valued
        assert_eq!(result.dtype(), numr::dtype::DType::F64);

        // Check dimensions
        let freq_bins = 64 / 2 + 1;
        let n_frames = (512 + 64 - 64) / 32 + 1;
        assert_eq!(result.shape(), &[n_frames, freq_bins]);

        // All values should be non-negative (power)
        let data: Vec<f64> = result.to_vec();
        for val in data {
            assert!(val >= 0.0);
        }
    }
}
