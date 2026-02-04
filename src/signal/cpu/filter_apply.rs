//! CPU-only implementations of IIR filter application algorithms.
//!
//! # Why CPU-Only?
//!
//! IIR (Infinite Impulse Response) filters are **inherently sequential** due to
//! their recurrence relation:
//!
//! ```text
//! y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...
//! ```
//!
//! Each output y[n] depends on the previous output y[n-1]. This data dependency
//! makes parallelization impossible - GPU acceleration provides ZERO benefit.
//!
//! Therefore, these implementations are **explicitly CPU-only**:
//! - No GPU↔CPU transfers (data stays on CPU)
//! - No generic `R: Runtime` abstraction
//! - Direct `CpuRuntime` types throughout
//!
//! For GPU users who need IIR filtering, transfer data to CPU explicitly,
//! call these functions, then transfer back if needed.

use crate::signal::filter::types::SosFilter;
use crate::signal::traits::filter_apply::{
    FilterApplicationAlgorithms, LfilterResult, PadType, SosfiltResult,
};
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Trait Implementation
// ============================================================================

impl FilterApplicationAlgorithms<CpuRuntime> for CpuClient {
    fn lfilter(
        &self,
        b: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        zi: Option<&Tensor<CpuRuntime>>,
    ) -> Result<LfilterResult<CpuRuntime>> {
        lfilter_impl(b, a, x, zi)
    }

    fn filtfilt(
        &self,
        b: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        padtype: Option<PadType>,
        padlen: Option<usize>,
    ) -> Result<Tensor<CpuRuntime>> {
        filtfilt_impl(b, a, x, padtype, padlen)
    }

    fn sosfilt(
        &self,
        sos: &SosFilter<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        zi: Option<&Tensor<CpuRuntime>>,
    ) -> Result<SosfiltResult<CpuRuntime>> {
        sosfilt_impl(sos, x, zi)
    }

    fn sosfiltfilt(
        &self,
        sos: &SosFilter<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        padtype: Option<PadType>,
        padlen: Option<usize>,
    ) -> Result<Tensor<CpuRuntime>> {
        sosfiltfilt_impl(sos, x, padtype, padlen)
    }
}

// ============================================================================
// Implementation Functions (CPU-only, not generic)
// ============================================================================

/// Apply IIR/FIR filter using Direct Form II transposed.
fn lfilter_impl(
    b: &Tensor<CpuRuntime>,
    a: &Tensor<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    zi: Option<&Tensor<CpuRuntime>>,
) -> Result<LfilterResult<CpuRuntime>> {
    // Validate inputs
    if b.ndim() != 1 || a.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "b/a",
            reason: "Filter coefficients must be 1D".to_string(),
        });
    }

    let nb = b.shape()[0];
    let na = a.shape()[0];
    let nfilt = nb.max(na);

    if nfilt == 0 {
        return Err(Error::InvalidArgument {
            arg: "b/a",
            reason: "Filter coefficients cannot be empty".to_string(),
        });
    }

    // Get coefficient data (CPU memory only - no transfer)
    let b_data: Vec<f64> = b.to_vec();
    let a_data: Vec<f64> = a.to_vec();

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

    // Get input data (CPU memory only - no transfer)
    let x_data: Vec<f64> = x.to_vec();
    let n_samples = x_data.len();

    // Initialize state
    let state_len = nfilt - 1;
    let mut z = if let Some(zi_tensor) = zi {
        let zi_data: Vec<f64> = zi_tensor.to_vec();
        if zi_data.len() != state_len {
            return Err(Error::InvalidArgument {
                arg: "zi",
                reason: format!("Initial state must have length {}", state_len),
            });
        }
        zi_data
    } else {
        vec![0.0; state_len]
    };

    // Apply filter using Direct Form II Transposed (sequential by necessity)
    let mut y = Vec::with_capacity(n_samples);

    for &xn in &x_data {
        // Output: y[n] = b[0]*x[n] + z[0]
        let yn = b_pad[0] * xn + if state_len > 0 { z[0] } else { 0.0 };
        y.push(yn);

        // Update state
        for i in 0..state_len {
            let b_term = if i + 1 < nfilt {
                b_pad[i + 1] * xn
            } else {
                0.0
            };
            let a_term = if i + 1 < nfilt {
                a_pad[i + 1] * yn
            } else {
                0.0
            };
            let z_term = if i + 1 < state_len { z[i + 1] } else { 0.0 };
            z[i] = b_term - a_term + z_term;
        }
    }

    let device = x.device();
    Ok(LfilterResult {
        y: Tensor::from_slice(&y, &[n_samples], device),
        zf: Tensor::from_slice(&z, &[state_len], device),
    })
}

/// Zero-phase digital filtering (forward-backward).
fn filtfilt_impl(
    b: &Tensor<CpuRuntime>,
    a: &Tensor<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    padtype: Option<PadType>,
    padlen: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    let padtype = padtype.unwrap_or_default();
    let nb = b.shape()[0];
    let na = a.shape()[0];
    let padlen = padlen.unwrap_or(3 * nb.max(na));

    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();

    if n <= padlen {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("Input must be longer than padlen ({})", padlen),
        });
    }

    // Pad the signal
    let padded = match padtype {
        PadType::None => x_data.clone(),
        PadType::Odd => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for i in (1..=padlen).rev() {
                result.push(2.0 * x_data[0] - x_data[i]);
            }
            result.extend_from_slice(&x_data);
            for i in 1..=padlen {
                result.push(2.0 * x_data[n - 1] - x_data[n - 1 - i]);
            }
            result
        }
        PadType::Even => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for i in (1..=padlen).rev() {
                result.push(x_data[i]);
            }
            result.extend_from_slice(&x_data);
            for i in 1..=padlen {
                result.push(x_data[n - 1 - i]);
            }
            result
        }
        PadType::Constant => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for _ in 0..padlen {
                result.push(x_data[0]);
            }
            result.extend_from_slice(&x_data);
            for _ in 0..padlen {
                result.push(x_data[n - 1]);
            }
            result
        }
    };

    let device = x.device();
    let x_padded = Tensor::from_slice(&padded, &[padded.len()], device);

    // Forward filter
    let forward = lfilter_impl(b, a, &x_padded, None)?;

    // Reverse
    let y1: Vec<f64> = forward.y.to_vec();
    let y1_rev: Vec<f64> = y1.into_iter().rev().collect();
    let y1_rev_tensor = Tensor::from_slice(&y1_rev, &[y1_rev.len()], device);

    // Backward filter
    let backward = lfilter_impl(b, a, &y1_rev_tensor, None)?;

    // Reverse again
    let y2: Vec<f64> = backward.y.to_vec();
    let y2_rev: Vec<f64> = y2.into_iter().rev().collect();

    // Remove padding
    let start = if padtype == PadType::None { 0 } else { padlen };
    let end = if padtype == PadType::None {
        n
    } else {
        padlen + n
    };

    let output: Vec<f64> = y2_rev[start..end].to_vec();
    Ok(Tensor::from_slice(&output, &[n], device))
}

/// Apply filter in second-order sections form.
fn sosfilt_impl(
    sos: &SosFilter<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    zi: Option<&Tensor<CpuRuntime>>,
) -> Result<SosfiltResult<CpuRuntime>> {
    let n_sections = sos.num_sections();
    let device = x.device();

    if n_sections == 0 {
        return Ok(SosfiltResult {
            y: x.clone(),
            zf: Tensor::zeros(&[0, 2], x.dtype(), device),
        });
    }

    // Get SOS coefficients
    let sos_data: Vec<f64> = sos.sections.to_vec();

    // Get input data
    let x_data: Vec<f64> = x.to_vec();
    let n_samples = x_data.len();

    // Initialize state [n_sections, 2]
    let mut z: Vec<Vec<f64>> = if let Some(zi_tensor) = zi {
        let zi_data: Vec<f64> = zi_tensor.to_vec();
        if zi_data.len() != n_sections * 2 {
            return Err(Error::InvalidArgument {
                arg: "zi",
                reason: format!("Initial state must have shape [{}, 2]", n_sections),
            });
        }
        (0..n_sections)
            .map(|i| vec![zi_data[i * 2], zi_data[i * 2 + 1]])
            .collect()
    } else {
        vec![vec![0.0, 0.0]; n_sections]
    };

    // Process signal through each section (sequential by necessity)
    let mut y = x_data.clone();

    for (section_idx, z_sec) in z.iter_mut().enumerate() {
        let offset = section_idx * 6;
        let b0 = sos_data[offset];
        let b1 = sos_data[offset + 1];
        let b2 = sos_data[offset + 2];
        let a0 = sos_data[offset + 3];
        let a1 = sos_data[offset + 4];
        let a2 = sos_data[offset + 5];

        // Normalize by a0
        let b0 = b0 / a0;
        let b1 = b1 / a0;
        let b2 = b2 / a0;
        let a1 = a1 / a0;
        let a2 = a2 / a0;
        let mut new_y = Vec::with_capacity(n_samples);

        for &xn in &y {
            // Output
            let yn = b0 * xn + z_sec[0];
            new_y.push(yn);

            // Update state
            let z0_new = b1 * xn - a1 * yn + z_sec[1];
            let z1_new = b2 * xn - a2 * yn;
            z_sec[0] = z0_new;
            z_sec[1] = z1_new;
        }

        y = new_y;
    }

    // Flatten final state
    let zf_flat: Vec<f64> = z.into_iter().flatten().collect();

    Ok(SosfiltResult {
        y: Tensor::from_slice(&y, &[n_samples], device),
        zf: Tensor::from_slice(&zf_flat, &[n_sections, 2], device),
    })
}

/// Zero-phase filtering using SOS.
fn sosfiltfilt_impl(
    sos: &SosFilter<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    padtype: Option<PadType>,
    padlen: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    let padtype = padtype.unwrap_or_default();
    let padlen = padlen.unwrap_or(3 * sos.order());

    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();

    if n <= padlen {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("Input must be longer than padlen ({})", padlen),
        });
    }

    // Pad the signal
    let padded = match padtype {
        PadType::None => x_data.clone(),
        PadType::Odd => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for i in (1..=padlen).rev() {
                result.push(2.0 * x_data[0] - x_data[i]);
            }
            result.extend_from_slice(&x_data);
            for i in 1..=padlen {
                result.push(2.0 * x_data[n - 1] - x_data[n - 1 - i]);
            }
            result
        }
        PadType::Even => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for i in (1..=padlen).rev() {
                result.push(x_data[i]);
            }
            result.extend_from_slice(&x_data);
            for i in 1..=padlen {
                result.push(x_data[n - 1 - i]);
            }
            result
        }
        PadType::Constant => {
            let mut result = Vec::with_capacity(n + 2 * padlen);
            for _ in 0..padlen {
                result.push(x_data[0]);
            }
            result.extend_from_slice(&x_data);
            for _ in 0..padlen {
                result.push(x_data[n - 1]);
            }
            result
        }
    };

    let device = x.device();
    let x_padded = Tensor::from_slice(&padded, &[padded.len()], device);

    // Forward filter
    let forward = sosfilt_impl(sos, &x_padded, None)?;

    // Reverse
    let y1: Vec<f64> = forward.y.to_vec();
    let y1_rev: Vec<f64> = y1.into_iter().rev().collect();
    let y1_rev_tensor = Tensor::from_slice(&y1_rev, &[y1_rev.len()], device);

    // Backward filter
    let backward = sosfilt_impl(sos, &y1_rev_tensor, None)?;

    // Reverse again
    let y2: Vec<f64> = backward.y.to_vec();
    let y2_rev: Vec<f64> = y2.into_iter().rev().collect();

    // Remove padding
    let start = if padtype == PadType::None { 0 } else { padlen };
    let end = if padtype == PadType::None {
        n
    } else {
        padlen + n
    };

    let output: Vec<f64> = y2_rev[start..end].to_vec();
    Ok(Tensor::from_slice(&output, &[n], device))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::filter::{FilterOutput, FilterType, IirDesignAlgorithms};
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_lfilter_fir() {
        let (client, device) = setup();

        // Simple moving average (FIR)
        let b = Tensor::<CpuRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);

        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[8],
            &device,
        );

        let result = client.lfilter(&b, &a, &x, None).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // Expected: moving average with initial transient
        assert_eq!(y.len(), 8);
        // After transient, should be average of 4 samples
        assert!((y[3] - 2.5).abs() < 1e-10); // (1+2+3+4)/4
        assert!((y[7] - 6.5).abs() < 1e-10); // (5+6+7+8)/4
    }

    #[test]
    fn test_lfilter_iir() {
        let (client, device) = setup();

        // Simple first-order IIR: y[n] = x[n] + 0.5*y[n-1]
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device);

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0], &[5], &device);

        let result = client.lfilter(&b, &a, &x, None).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // Impulse response: 1, 0.5, 0.25, 0.125, 0.0625
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 0.5).abs() < 1e-10);
        assert!((y[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_filtfilt() {
        let (client, device) = setup();

        // Design a lowpass filter
        let result = client
            .butter(2, &[0.1], FilterType::Lowpass, FilterOutput::Ba, &device)
            .unwrap();
        let tf = result.as_ba().unwrap();

        // Create a noisy signal
        let mut signal = Vec::with_capacity(100);
        for i in 0..100 {
            signal.push((i as f64 * 0.1).sin() + 0.1 * ((i as f64 * 2.5).sin()));
        }
        let x = Tensor::<CpuRuntime>::from_slice(&signal, &[100], &device);

        let y = client.filtfilt(&tf.b, &tf.a, &x, None, None).unwrap();

        assert_eq!(y.shape(), &[100]);
    }

    #[test]
    fn test_sosfilt() {
        let (client, device) = setup();

        // Design a filter in SOS form
        let result = client
            .butter(4, &[0.2], FilterType::Lowpass, FilterOutput::Sos, &device)
            .unwrap();
        let sos = result.as_sos().unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[8],
            &device,
        );

        let result = client.sosfilt(sos, &x, None).unwrap();

        assert_eq!(result.y.shape(), &[8]);
        assert_eq!(result.zf.shape(), &[2, 2]); // 2 sections, 2 states each
    }
}
