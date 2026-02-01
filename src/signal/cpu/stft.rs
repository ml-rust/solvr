//! CPU implementation of STFT and inverse STFT algorithms.

use crate::signal::impl_generic::{istft_impl, stft_impl};
use crate::signal::traits::stft::StftAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl StftAlgorithms<CpuRuntime> for CpuClient {
    fn stft(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CpuRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
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
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_stft_basic() {
        let (client, device) = setup();

        // Create a simple signal
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[256], &device);

        let result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        // Output should be complex with shape [n_frames, freq_bins]
        let freq_bins = 64 / 2 + 1; // 33
        let n_frames = (256 + 64 - 64) / 16 + 1; // 17

        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }

    #[test]
    fn test_istft_reconstruction() {
        let (client, device) = setup();

        // Create a signal
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[256], &device);

        // STFT
        let stft_result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        // ISTFT
        let reconstructed = client
            .istft(&stft_result, Some(16), None, true, Some(256), false)
            .unwrap();

        assert_eq!(reconstructed.shape(), &[256]);

        // Check reconstruction is close to original
        let recon_data: Vec<f64> = reconstructed.to_vec();
        for (i, (&orig, &recon)) in signal.iter().zip(recon_data.iter()).enumerate() {
            let err = (orig - recon).abs();
            // Skip edges which have boundary effects
            if i > 32 && i < 224 {
                assert!(
                    err < 0.1,
                    "Reconstruction error at {}: {} vs {}",
                    i,
                    orig,
                    recon
                );
            }
        }
    }
}
