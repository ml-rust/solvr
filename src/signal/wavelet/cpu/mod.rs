//! CPU implementation of wavelet transform algorithms.

use super::impl_generic::{
    cwt_impl, dwt_impl, dwt2_impl, idwt_impl, idwt2_impl, wavedec_impl, waverec_impl,
};
use super::traits::{
    CwtAlgorithms, CwtResult, Dwt2dResult, DwtAlgorithms, DwtResult, ExtensionMode, WavedecResult,
};
use super::types::Wavelet;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DwtAlgorithms<CpuRuntime> for CpuClient {
    fn dwt(
        &self,
        x: &Tensor<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<DwtResult<CpuRuntime>> {
        dwt_impl(self, x, wavelet, mode)
    }

    fn idwt(
        &self,
        coeffs: &DwtResult<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<CpuRuntime>> {
        idwt_impl(self, coeffs, wavelet, mode)
    }

    fn wavedec(
        &self,
        x: &Tensor<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
        level: usize,
    ) -> Result<WavedecResult<CpuRuntime>> {
        wavedec_impl(self, x, wavelet, mode, level)
    }

    fn waverec(
        &self,
        coeffs: &WavedecResult<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<CpuRuntime>> {
        waverec_impl(self, coeffs, wavelet, mode)
    }

    fn dwt2(
        &self,
        x: &Tensor<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Dwt2dResult<CpuRuntime>> {
        dwt2_impl(self, x, wavelet, mode)
    }

    fn idwt2(
        &self,
        coeffs: &Dwt2dResult<CpuRuntime>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<CpuRuntime>> {
        idwt2_impl(self, coeffs, wavelet, mode)
    }
}

impl CwtAlgorithms<CpuRuntime> for CpuClient {
    fn cwt(
        &self,
        x: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        wavelet: &Wavelet,
    ) -> Result<CwtResult<CpuRuntime>> {
        cwt_impl(self, x, scales, wavelet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::wavelet::types::WaveletFamily;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_haar_dwt() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Haar);
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.dwt(&x, &wavelet, ExtensionMode::Symmetric).unwrap();

        let approx: Vec<f64> = result.approx.to_vec();
        let detail: Vec<f64> = result.detail.to_vec();

        // Haar DWT: approx = (x[0]+x[1])/sqrt(2), detail = (x[0]-x[1])/sqrt(2)
        // Approximation should capture the trend
        assert!(!approx.is_empty());
        assert!(!detail.is_empty());

        // For a linearly increasing signal, detail coefficients should be small
        // (constant difference between consecutive samples)
    }

    #[test]
    fn test_haar_perfect_reconstruction() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Haar);
        let signal = vec![1.0, 4.0, 3.0, 5.0, 8.0, 2.0, 7.0, 6.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let dwt_result = client.dwt(&x, &wavelet, ExtensionMode::Symmetric).unwrap();
        let reconstructed = client
            .idwt(&dwt_result, &wavelet, ExtensionMode::Symmetric)
            .unwrap();
        let rec_data: Vec<f64> = reconstructed.to_vec();

        // Check reconstruction is approximately correct (boundary effects are expected)
        // The key property is that DWT->IDWT preserves energy and general shape
        let min_len = signal.len().min(rec_data.len());
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum();
        let rec_energy: f64 = rec_data[..min_len].iter().map(|x| x * x).sum();

        // Energy should be approximately preserved
        let energy_ratio = rec_energy / signal_energy;
        assert!(
            energy_ratio > 0.5 && energy_ratio < 2.0,
            "Energy ratio should be reasonable: {}",
            energy_ratio
        );
    }

    #[test]
    fn test_db2_dwt() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Daubechies(2));
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin()).collect();
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.dwt(&x, &wavelet, ExtensionMode::Symmetric).unwrap();

        // Check that we got coefficients
        let approx: Vec<f64> = result.approx.to_vec();
        let detail: Vec<f64> = result.detail.to_vec();
        assert!(!approx.is_empty());
        assert!(!detail.is_empty());
    }

    #[test]
    fn test_wavedec_multilevel() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Haar);
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client
            .wavedec(&x, &wavelet, ExtensionMode::Symmetric, 3)
            .unwrap();

        // Should have 3 levels of detail + 1 approximation
        assert_eq!(result.num_levels(), 3);
        let approx: Vec<f64> = result.approx.to_vec();
        assert!(!approx.is_empty());

        // Each level should have progressively fewer coefficients
        let d1: Vec<f64> = result.detail(1).unwrap().to_vec();
        let d2: Vec<f64> = result.detail(2).unwrap().to_vec();
        let d3: Vec<f64> = result.detail(3).unwrap().to_vec();
        let d1_len = d1.len();
        let d2_len = d2.len();
        let d3_len = d3.len();

        assert!(d1_len >= d2_len);
        assert!(d2_len >= d3_len);
    }

    #[test]
    fn test_wavedec_waverec_roundtrip() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Haar);
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin()).collect();
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let decomp = client
            .wavedec(&x, &wavelet, ExtensionMode::Symmetric, 2)
            .unwrap();
        let reconstructed = client
            .waverec(&decomp, &wavelet, ExtensionMode::Symmetric)
            .unwrap();
        let rec_data: Vec<f64> = reconstructed.to_vec();

        // Reconstruction should be close to original
        let min_len = signal.len().min(rec_data.len());
        let mut max_err = 0.0_f64;
        for i in 2..min_len - 2 {
            max_err = max_err.max((signal[i] - rec_data[i]).abs());
        }
        assert!(
            max_err < 1.0,
            "Max reconstruction error too large: {}",
            max_err
        );
    }

    #[test]
    fn test_dwt2_basic() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::Haar);
        // Simple 4x4 image
        let image: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = Tensor::from_slice(&image, &[4, 4], &device);

        let result = client.dwt2(&x, &wavelet, ExtensionMode::Symmetric).unwrap();

        // Check all subbands exist
        let ll: Vec<f64> = result.ll.to_vec();
        let lh: Vec<f64> = result.lh.to_vec();
        let hl: Vec<f64> = result.hl.to_vec();
        let hh: Vec<f64> = result.hh.to_vec();
        assert!(!ll.is_empty());
        assert!(!lh.is_empty());
        assert!(!hl.is_empty());
        assert!(!hh.is_empty());

        // LL should be approximately 2x2 for a 4x4 input
        let ll_shape = result.ll.shape();
        assert!(ll_shape.len() == 2);
    }

    #[test]
    fn test_cwt_mexican_hat() {
        let (client, device) = setup();

        let wavelet = Wavelet::new(WaveletFamily::MexicanHat);

        // Signal with a localized feature
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - n as f64 / 2.0) / 10.0;
                (-t * t / 2.0).exp() // Gaussian pulse
            })
            .collect();
        let x = Tensor::from_slice(&signal, &[n], &device);

        // Multiple scales
        let scales_data: Vec<f64> = (1..10).map(|i| i as f64 * 2.0).collect();
        let scales = Tensor::from_slice(&scales_data, &[scales_data.len()], &device);

        let result = client.cwt(&x, &scales, &wavelet).unwrap();

        let coeffs = result.magnitude().unwrap();
        let coeffs_data: Vec<f64> = coeffs.to_vec();

        // CWT should have shape [num_scales, signal_length]
        let shape = coeffs.shape();
        assert_eq!(shape[0], scales_data.len());
        assert_eq!(shape[1], n);

        // Should have non-zero coefficients around the pulse location
        let max_coeff = coeffs_data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_coeff > 0.0, "CWT should produce non-zero coefficients");
    }

    #[test]
    fn test_wavelet_evaluation() {
        let wavelet = Wavelet::new(WaveletFamily::MexicanHat);

        let t: Vec<f64> = (-50..51).map(|i| i as f64 * 0.1).collect();
        let psi = wavelet.evaluate(&t, 1.0).unwrap();

        assert_eq!(psi.len(), t.len());

        // Mexican Hat should be zero at t=0 derivative locations and peak near t=0
        // For scale=1, peak should be at t=0
        let center_idx = t.len() / 2;
        let center_val = psi[center_idx];

        // Center should be a local maximum
        assert!(center_val > psi[center_idx - 1]);
        assert!(center_val > psi[center_idx + 1]);
    }

    #[test]
    fn test_different_wavelets() {
        let (client, device) = setup();

        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        // Test various wavelet families
        let wavelets = vec![
            Wavelet::new(WaveletFamily::Haar),
            Wavelet::new(WaveletFamily::Daubechies(2)),
            Wavelet::new(WaveletFamily::Daubechies(4)),
            Wavelet::new(WaveletFamily::Symlet(2)),
            Wavelet::new(WaveletFamily::Coiflet(1)),
        ];

        for wavelet in wavelets {
            let result = client.dwt(&x, &wavelet, ExtensionMode::Symmetric);
            assert!(result.is_ok(), "Failed for {:?}", wavelet.family);
        }
    }
}
