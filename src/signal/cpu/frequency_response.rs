//! CPU implementation of frequency response algorithms.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::types::SosFilter;
use crate::signal::impl_generic::{freqz_impl, group_delay_impl, sosfreqz_impl};
use crate::signal::traits::frequency_response::{
    FrequencyResponseAlgorithms, FreqzResult, FreqzSpec,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl FrequencyResponseAlgorithms<CpuRuntime> for CpuClient {
    fn freqz(
        &self,
        b: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        worN: FreqzSpec<CpuRuntime>,
        whole: bool,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<FreqzResult<CpuRuntime>> {
        freqz_impl(self, b, a, worN, whole, device)
    }

    fn sosfreqz(
        &self,
        sos: &SosFilter<CpuRuntime>,
        worN: FreqzSpec<CpuRuntime>,
        whole: bool,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<FreqzResult<CpuRuntime>> {
        sosfreqz_impl(self, sos, worN, whole, device)
    }

    fn group_delay(
        &self,
        b: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        w: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        group_delay_impl(self, b, a, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use std::f64::consts::PI;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_freqz_lowpass() {
        let (client, device) = setup();

        // Simple 3-tap lowpass FIR: [0.25, 0.5, 0.25]
        let b = Tensor::from_slice(&[0.25, 0.5, 0.25], &[3], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        let result = client
            .freqz(&b, &a, FreqzSpec::NumPoints(128), false, &device)
            .unwrap();

        // Check dimensions
        let w: Vec<f64> = result.w.to_vec();
        assert_eq!(w.len(), 128);

        // Check frequency range (0 to pi)
        assert!((w[0] - 0.0).abs() < 1e-10);
        assert!((w[127] - PI * 127.0 / 128.0).abs() < 1e-10);

        // Get magnitude response
        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At DC (omega=0), magnitude should be 1.0
        assert!((mag_data[0] - 1.0).abs() < 1e-10);

        // At Nyquist (omega=pi), magnitude should be 0 for this filter
        // Since we're evaluating at 128 points, the last point is near pi
        assert!(mag_data[127] < 0.1);
    }

    #[test]
    fn test_freqz_allpass() {
        let (client, device) = setup();

        // Unity gain filter
        let b = Tensor::from_slice(&[1.0], &[1], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        let result = client
            .freqz(&b, &a, FreqzSpec::NumPoints(64), false, &device)
            .unwrap();

        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // Unity gain at all frequencies
        for m in &mag_data {
            assert!((*m - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_freqz_specific_frequencies() {
        let (client, device) = setup();

        // Test at specific frequencies
        let b = Tensor::from_slice(&[1.0, -1.0], &[2], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        // Evaluate at 0, pi/2, pi
        let w_test = Tensor::from_slice(&[0.0, PI / 2.0, PI], &[3], &device);
        let result = client
            .freqz(&b, &a, FreqzSpec::Frequencies(w_test), false, &device)
            .unwrap();

        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At DC: H(1) = 1 - 1 = 0
        assert!(mag_data[0].abs() < 1e-10);

        // At pi/2: |1 - e^{-j*pi/2}| = |1 + j| = sqrt(2)
        assert!((mag_data[1] - 2.0_f64.sqrt()).abs() < 1e-10);

        // At Nyquist: H(-1) = 1 - (-1) = 2
        assert!((mag_data[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_freqz_whole() {
        let (client, device) = setup();

        let b = Tensor::from_slice(&[1.0], &[1], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        let result = client
            .freqz(&b, &a, FreqzSpec::NumPoints(64), true, &device)
            .unwrap();

        let w: Vec<f64> = result.w.to_vec();

        // Should span 0 to 2*pi (exclusive of 2*pi)
        assert!((w[0] - 0.0).abs() < 1e-10);
        assert!((w[63] - 2.0 * PI * 63.0 / 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_sosfreqz_single_section() {
        let (client, device) = setup();

        // Single section: [b0, b1, b2, a0, a1, a2] = [0.25, 0.5, 0.25, 1, 0, 0]
        let sections = Tensor::from_slice(&[0.25, 0.5, 0.25, 1.0, 0.0, 0.0], &[1, 6], &device);
        let sos = SosFilter { sections };

        let sos_result = client
            .sosfreqz(&sos, FreqzSpec::NumPoints(64), false, &device)
            .unwrap();

        // Compare with equivalent tf form
        let b = Tensor::from_slice(&[0.25, 0.5, 0.25], &[3], &device);
        let a = Tensor::from_slice(&[1.0, 0.0, 0.0], &[3], &device);
        let tf_result = client
            .freqz(&b, &a, FreqzSpec::NumPoints(64), false, &device)
            .unwrap();

        let sos_mag = sos_result.magnitude().unwrap();
        let tf_mag = tf_result.magnitude().unwrap();

        let sos_data: Vec<f64> = sos_mag.to_vec();
        let tf_data: Vec<f64> = tf_mag.to_vec();

        for (s, t) in sos_data.iter().zip(tf_data.iter()) {
            assert!((*s - *t).abs() < 1e-10, "SOS: {}, TF: {}", s, t);
        }
    }

    #[test]
    fn test_sosfreqz_cascaded() {
        let (client, device) = setup();

        // Two identical sections
        let sections = Tensor::from_slice(
            &[
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Section 1: unity
                0.5, 0.5, 0.0, 1.0, 0.0, 0.0, // Section 2: simple lowpass
            ],
            &[2, 6],
            &device,
        );
        let sos = SosFilter { sections };

        let result = client
            .sosfreqz(&sos, FreqzSpec::NumPoints(32), false, &device)
            .unwrap();

        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At DC: both sections pass through, magnitude = 1.0
        assert!((mag_data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_group_delay_fir() {
        let (client, device) = setup();

        // Linear phase FIR with N=5 taps has constant group delay = (N-1)/2 = 2
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 2.0, 1.0], &[5], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        // Test at a few frequencies
        let w = Tensor::from_slice(&[0.1, 0.5, 1.0, 2.0], &[4], &device);

        let tau = client.group_delay(&b, &a, &w).unwrap();
        let tau_data: Vec<f64> = tau.to_vec();

        // For symmetric FIR filter, group delay should be (N-1)/2 = 2
        for t in &tau_data {
            assert!((*t - 2.0).abs() < 0.1, "Expected ~2.0, got {}", t);
        }
    }

    #[test]
    fn test_magnitude_db() {
        let (client, device) = setup();

        // Filter with known gain
        let b = Tensor::from_slice(&[2.0], &[1], &device); // Gain of 2
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        let result = client
            .freqz(&b, &a, FreqzSpec::NumPoints(10), false, &device)
            .unwrap();

        let mag_db = result.magnitude_db().unwrap();
        let db_data: Vec<f64> = mag_db.to_vec();

        // 20*log10(2) ≈ 6.02 dB
        let expected_db = 20.0 * 2.0_f64.log10();
        for db in &db_data {
            assert!((*db - expected_db).abs() < 1e-10);
        }
    }

    #[test]
    fn test_phase_response() {
        let (client, device) = setup();

        // Simple delay: H(z) = z^{-1}, phase = -omega
        let b = Tensor::from_slice(&[0.0, 1.0], &[2], &device);
        let a = Tensor::from_slice(&[1.0], &[1], &device);

        let w_test = Tensor::from_slice(&[0.0, PI / 4.0, PI / 2.0, PI], &[4], &device);
        let result = client
            .freqz(&b, &a, FreqzSpec::Frequencies(w_test), false, &device)
            .unwrap();

        let phase = result.phase().unwrap();
        let phase_data: Vec<f64> = phase.to_vec();

        // Phase should be -omega (or wrapped equivalent)
        assert!(phase_data[0].abs() < 1e-10); // At DC: 0
        assert!((phase_data[1] - (-PI / 4.0)).abs() < 1e-10);
        assert!((phase_data[2] - (-PI / 2.0)).abs() < 1e-10);
        // At pi, phase is -pi (or +pi due to wrapping)
        assert!((phase_data[3].abs() - PI).abs() < 1e-10);
    }
}
