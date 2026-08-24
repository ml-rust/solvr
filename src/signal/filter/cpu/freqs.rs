//! CPU implementation of analog frequency response.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::impl_generic::freqs_impl;
use crate::signal::filter::traits::analog_response::{AnalogResponseAlgorithms, FreqsResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AnalogResponseAlgorithms<CpuRuntime> for CpuClient {
    fn freqs(
        &self,
        b: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        worN: &Tensor<CpuRuntime>,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<FreqsResult<CpuRuntime>> {
        freqs_impl(self, b, a, worN, device)
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
    fn test_freqs_first_order_lowpass() {
        let (client, device) = setup();

        // First-order analog lowpass: H(s) = 1 / (s + 1)
        // At ω = 0: H = 1
        // At ω = 1: H = 1/(1+j) = (1-j)/2, |H| = 1/√2 ≈ 0.707
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device).unwrap();

        let w =
            Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0, 10.0, 100.0], &[4], &device).unwrap();

        let result = client.freqs(&b, &a, &w, &device).unwrap();
        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At ω=0, |H| = 1
        assert!((mag_data[0] - 1.0).abs() < 1e-10);

        // At ω=1, |H| = 1/√2 ≈ 0.707
        assert!((mag_data[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);

        // As ω increases, |H| → 0
        assert!(mag_data[3] < 0.02);
    }

    #[test]
    fn test_freqs_second_order() {
        let (client, device) = setup();

        // Second-order analog lowpass (Butterworth): H(s) = 1 / (s² + √2·s + 1)
        // At ω = 0: H = 1
        // At ω = 1: |H| = 1/√2
        let sqrt2 = 2.0_f64.sqrt();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, sqrt2, 1.0], &[3], &device).unwrap();

        let w = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0], &[2], &device).unwrap();

        let result = client.freqs(&b, &a, &w, &device).unwrap();
        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At ω=0, |H| = 1
        assert!((mag_data[0] - 1.0).abs() < 1e-10);

        // At ω=1, |H| = 1/√2 for Butterworth
        assert!((mag_data[1] - 1.0 / sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_freqs_bandpass() {
        let (client, device) = setup();

        // Bandpass around ω₀=1: H(s) = s / (s² + s + 1)
        // At ω = 1: passes (resonance)
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0], &[2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device).unwrap();

        let w = Tensor::<CpuRuntime>::from_slice(&[0.01f64, 1.0, 100.0], &[3], &device).unwrap();

        let result = client.freqs(&b, &a, &w, &device).unwrap();
        let mag = result.magnitude().unwrap();
        let mag_data: Vec<f64> = mag.to_vec();

        // At very low ω, magnitude is small
        assert!(mag_data[0] < 0.02);

        // At ω=1, magnitude is maximum (Q-dependent, but should be reasonable)
        assert!(mag_data[1] > 0.5);

        // At very high ω, magnitude decreases
        assert!(mag_data[2] < 0.02);
    }

    #[test]
    fn test_freqs_phase() {
        let (client, device) = setup();

        // First-order lowpass: H(s) = 1 / (s + 1)
        // Phase at ω=0: 0°
        // Phase at ω=1: -45° = -π/4
        // Phase at ω→∞: -90° = -π/2
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device).unwrap();

        let w = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0, 1000.0], &[3], &device).unwrap();

        let result = client.freqs(&b, &a, &w, &device).unwrap();
        let phase = result.phase().unwrap();
        let phase_data: Vec<f64> = phase.to_vec();

        // At ω=0, phase = 0
        assert!(phase_data[0].abs() < 1e-10);

        // At ω=1, phase = -π/4
        assert!((phase_data[1] + PI / 4.0).abs() < 1e-10);

        // At ω→∞, phase → -π/2
        assert!((phase_data[2] + PI / 2.0).abs() < 0.01);
    }
}
