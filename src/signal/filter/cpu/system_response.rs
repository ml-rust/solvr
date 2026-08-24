//! CPU-only implementation of system response algorithms.
//!
//! # Why CPU-Only?
//!
//! System response computation (impulse, step) uses IIR filtering which is
//! **inherently sequential** due to the recurrence relation:
//!
//! ```text
//! y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...
//! ```
//!
//! Each output y[n] depends on the previous output y[n-1]. This data dependency
//! makes parallelization impossible - GPU acceleration provides ZERO benefit.

use crate::signal::filter::traits::system_response::{
    ImpulseResponse, StepResponse, SystemResponseAlgorithms,
};
use crate::signal::filter::types::TransferFunction;
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SystemResponseAlgorithms<CpuRuntime> for CpuClient {
    fn impulse(
        &self,
        tf: &TransferFunction<CpuRuntime>,
        n: usize,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<ImpulseResponse<CpuRuntime>> {
        impulse_impl(tf, n, device)
    }

    fn step(
        &self,
        tf: &TransferFunction<CpuRuntime>,
        n: usize,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<StepResponse<CpuRuntime>> {
        step_impl(tf, n, device)
    }
}

// ============================================================================
// Implementation Functions (CPU-only, not generic)
// ============================================================================

/// Compute impulse response by applying filter to impulse input.
fn impulse_impl(
    tf: &TransferFunction<CpuRuntime>,
    n: usize,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<ImpulseResponse<CpuRuntime>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "Number of samples must be > 0".to_string(),
        });
    }

    // Get filter coefficients (CPU memory - no transfer)
    let b_data: Vec<f64> = tf.b.to_vec();
    let a_data: Vec<f64> = tf.a.to_vec();

    let nb = b_data.len();
    let na = a_data.len();
    let nfilt = nb.max(na);

    // Normalize by a[0]
    let a0 = a_data[0];
    if a0.abs() < 1e-30 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "Leading denominator coefficient cannot be zero".to_string(),
        });
    }

    let b_norm: Vec<f64> = b_data.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a_data.iter().map(|&x| x / a0).collect();

    // Pad coefficients to same length
    let mut b_pad = vec![0.0; nfilt];
    let mut a_pad = vec![0.0; nfilt];
    b_pad[..nb].copy_from_slice(&b_norm);
    a_pad[..na].copy_from_slice(&a_norm);

    // Create impulse input: [1, 0, 0, 0, ...]
    let mut x = vec![0.0; n];
    x[0] = 1.0;

    // Apply filter using Direct Form II Transposed (sequential by necessity)
    let y = apply_filter(&b_pad, &a_pad, &x);

    // Create time indices
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();

    Ok(ImpulseResponse {
        t: Tensor::from_slice(&t, &[n], device)?,
        y: Tensor::from_slice(&y, &[n], device)?,
    })
}

/// Compute step response by applying filter to step input.
fn step_impl(
    tf: &TransferFunction<CpuRuntime>,
    n: usize,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<StepResponse<CpuRuntime>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "Number of samples must be > 0".to_string(),
        });
    }

    // Get filter coefficients (CPU memory - no transfer)
    let b_data: Vec<f64> = tf.b.to_vec();
    let a_data: Vec<f64> = tf.a.to_vec();

    let nb = b_data.len();
    let na = a_data.len();
    let nfilt = nb.max(na);

    // Normalize by a[0]
    let a0 = a_data[0];
    if a0.abs() < 1e-30 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "Leading denominator coefficient cannot be zero".to_string(),
        });
    }

    let b_norm: Vec<f64> = b_data.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a_data.iter().map(|&x| x / a0).collect();

    // Pad coefficients to same length
    let mut b_pad = vec![0.0; nfilt];
    let mut a_pad = vec![0.0; nfilt];
    b_pad[..nb].copy_from_slice(&b_norm);
    a_pad[..na].copy_from_slice(&a_norm);

    // Create step input: [1, 1, 1, 1, ...]
    let x = vec![1.0; n];

    // Apply filter using Direct Form II Transposed (sequential by necessity)
    let y = apply_filter(&b_pad, &a_pad, &x);

    // Create time indices
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();

    Ok(StepResponse {
        t: Tensor::from_slice(&t, &[n], device)?,
        y: Tensor::from_slice(&y, &[n], device)?,
    })
}

/// Apply IIR filter using Direct Form II Transposed.
///
/// This is inherently sequential - each output depends on the previous.
fn apply_filter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let nfilt = b.len();
    let state_len = nfilt - 1;
    let mut z = vec![0.0; state_len];
    let mut y = Vec::with_capacity(x.len());

    for &xn in x {
        // Output: y[n] = b[0]*x[n] + z[0]
        let yn = b[0] * xn + if state_len > 0 { z[0] } else { 0.0 };
        y.push(yn);

        // Update state
        for i in 0..state_len {
            let b_term = if i + 1 < nfilt { b[i + 1] * xn } else { 0.0 };
            let a_term = if i + 1 < nfilt { a[i + 1] * yn } else { 0.0 };
            let z_term = if i + 1 < state_len { z[i + 1] } else { 0.0 };
            z[i] = b_term - a_term + z_term;
        }
    }

    y
}

// ============================================================================
// Tests
// ============================================================================

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
    fn test_impulse_fir() {
        let (client, device) = setup();

        // FIR filter: moving average [0.25, 0.25, 0.25, 0.25]
        // Impulse response should be the filter coefficients
        let b =
            Tensor::<CpuRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.impulse(&tf, 6, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        assert_eq!(y.len(), 6);
        assert!((y[0] - 0.25).abs() < 1e-10);
        assert!((y[1] - 0.25).abs() < 1e-10);
        assert!((y[2] - 0.25).abs() < 1e-10);
        assert!((y[3] - 0.25).abs() < 1e-10);
        assert!(y[4].abs() < 1e-10);
        assert!(y[5].abs() < 1e-10);
    }

    #[test]
    fn test_impulse_iir() {
        let (client, device) = setup();

        // First-order IIR: y[n] = x[n] + 0.5*y[n-1]
        // H(z) = 1 / (1 - 0.5z^-1)
        // Impulse response: h[n] = 0.5^n
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.impulse(&tf, 5, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 0.5).abs() < 1e-10);
        assert!((y[2] - 0.25).abs() < 1e-10);
        assert!((y[3] - 0.125).abs() < 1e-10);
        assert!((y[4] - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn test_step_fir() {
        let (client, device) = setup();

        // FIR filter: moving average [0.25, 0.25, 0.25, 0.25]
        // Step response: cumulative sum of impulse response
        let b =
            Tensor::<CpuRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.step(&tf, 6, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        assert!((y[0] - 0.25).abs() < 1e-10);
        assert!((y[1] - 0.50).abs() < 1e-10);
        assert!((y[2] - 0.75).abs() < 1e-10);
        assert!((y[3] - 1.00).abs() < 1e-10);
        assert!((y[4] - 1.00).abs() < 1e-10); // Steady state
        assert!((y[5] - 1.00).abs() < 1e-10);
    }

    #[test]
    fn test_step_iir() {
        let (client, device) = setup();

        // First-order IIR: y[n] = x[n] + 0.5*y[n-1]
        // Step response converges to 1 / (1 - 0.5) = 2
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.step(&tf, 20, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // First few samples
        assert!((y[0] - 1.0).abs() < 1e-10); // 1
        assert!((y[1] - 1.5).abs() < 1e-10); // 1 + 0.5
        assert!((y[2] - 1.75).abs() < 1e-10); // 1 + 0.5 + 0.25

        // Should converge to 2
        assert!((y[19] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_impulse_time_indices() {
        let (client, device) = setup();

        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.impulse(&tf, 5, &device).unwrap();
        let t: Vec<f64> = result.t.to_vec();

        assert_eq!(t, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_impulse_second_order() {
        let (client, device) = setup();

        // Second-order system: damped oscillation
        // H(z) = 1 / (1 - 1.2z^-1 + 0.5z^-2)
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.2, 0.5], &[3], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let result = client.impulse(&tf, 10, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // First sample should be 1
        assert!((y[0] - 1.0).abs() < 1e-10);

        // Check it's a damped response (magnitude should decrease overall)
        assert!(y[9].abs() < y[0].abs());
    }
}
