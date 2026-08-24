//! CPU implementation of FIR filter design.
//!
//! FIR filter design is CPU-only because:
//! 1. These are design-time operations (run once), not runtime (per-sample)
//! 2. FIR tap counts are typically 31-255 (tiny data)
//! 3. Algorithms involve sequential operations (sinc computation, interpolation)
//! 4. GPU transfer overhead far exceeds computation time

// Allow indexed loops for filter coefficient computation
#![allow(clippy::needless_range_loop)]
// Allow manual div_ceil for clarity
#![allow(clippy::manual_div_ceil)]

use crate::signal::filter::traits::fir_design::{FirDesignAlgorithms, FirWindow};
use crate::signal::filter::types::FilterType;
use crate::window::WindowFunctions;
use numr::algorithm::fft::FftAlgorithms;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{BinaryOps, ComplexOps, ScalarOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::f64::consts::PI;

impl FirDesignAlgorithms<CpuRuntime> for CpuClient {
    fn firwin(
        &self,
        numtaps: usize,
        cutoff: &[f64],
        filter_type: FilterType,
        window: FirWindow,
        scale: bool,
        device: &CpuDevice,
    ) -> Result<Tensor<CpuRuntime>> {
        firwin_cpu(self, numtaps, cutoff, filter_type, window, scale, device)
    }

    fn firwin2(
        &self,
        numtaps: usize,
        freq: &[f64],
        gain: &[f64],
        antisymmetric: bool,
        window: FirWindow,
        device: &CpuDevice,
    ) -> Result<Tensor<CpuRuntime>> {
        firwin2_cpu(self, numtaps, freq, gain, antisymmetric, window, device)
    }

    fn minimum_phase(&self, h: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        minimum_phase_cpu(self, h)
    }
}

// ============================================================================
// CPU-only FIR design implementations
// ============================================================================

/// Design FIR filter using windowed sinc method (CPU-only).
fn firwin_cpu(
    client: &CpuClient,
    numtaps: usize,
    cutoff: &[f64],
    filter_type: FilterType,
    window: FirWindow,
    scale: bool,
    device: &CpuDevice,
) -> Result<Tensor<CpuRuntime>> {
    if numtaps == 0 {
        return Err(Error::InvalidArgument {
            arg: "numtaps",
            reason: "Number of taps must be > 0".to_string(),
        });
    }

    validate_fir_cutoff(cutoff, filter_type)?;

    let dtype = DType::F64;
    let alpha = (numtaps - 1) as f64 / 2.0;

    // Generate time indices centered at alpha
    let mut h = vec![0.0f64; numtaps];

    match filter_type {
        FilterType::Lowpass => {
            let fc = cutoff[0];
            for (i, hi) in h.iter_mut().enumerate() {
                let n = i as f64 - alpha;
                if n.abs() < 1e-10 {
                    *hi = 2.0 * fc;
                } else {
                    *hi = (2.0 * PI * fc * n).sin() / (PI * n);
                }
            }
        }
        FilterType::Highpass => {
            let fc = cutoff[0];
            for (i, hi) in h.iter_mut().enumerate() {
                let n = i as f64 - alpha;
                if n.abs() < 1e-10 {
                    *hi = 1.0 - 2.0 * fc;
                } else {
                    *hi = -(2.0 * PI * fc * n).sin() / (PI * n);
                }
            }
            // Spectral inversion for highpass
            for (i, hi) in h.iter_mut().enumerate() {
                if (i as f64 - alpha).abs() < 1e-10 {
                    *hi += 1.0;
                }
            }
        }
        FilterType::Bandpass => {
            let fc_low = cutoff[0];
            let fc_high = cutoff[1];
            for (i, hi) in h.iter_mut().enumerate() {
                let n = i as f64 - alpha;
                if n.abs() < 1e-10 {
                    *hi = 2.0 * (fc_high - fc_low);
                } else {
                    *hi = (2.0 * PI * fc_high * n).sin() / (PI * n)
                        - (2.0 * PI * fc_low * n).sin() / (PI * n);
                }
            }
        }
        FilterType::Bandstop => {
            let fc_low = cutoff[0];
            let fc_high = cutoff[1];
            for (i, hi) in h.iter_mut().enumerate() {
                let n = i as f64 - alpha;
                if n.abs() < 1e-10 {
                    *hi = 1.0 - 2.0 * (fc_high - fc_low);
                } else {
                    *hi = (2.0 * PI * fc_low * n).sin() / (PI * n)
                        - (2.0 * PI * fc_high * n).sin() / (PI * n);
                }
            }
            // Add impulse for bandstop
            let center = (numtaps - 1) / 2;
            h[center] += 1.0;
        }
    }

    // Apply window
    let h_tensor = Tensor::from_slice(&h, &[numtaps], device)?;
    let win = generate_window(client, numtaps, &window, dtype, device)?;
    let h_windowed = client.mul(&h_tensor, &win)?;

    // Scale for unity gain
    if scale {
        let gain = compute_gain(&h_windowed, cutoff, filter_type)?;
        if gain.abs() > 1e-10 {
            return client.mul_scalar(&h_windowed, 1.0 / gain);
        }
    }

    Ok(h_windowed)
}

/// Design FIR filter using frequency sampling method (CPU-only).
fn firwin2_cpu(
    client: &CpuClient,
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    antisymmetric: bool,
    window: FirWindow,
    device: &CpuDevice,
) -> Result<Tensor<CpuRuntime>> {
    if numtaps == 0 {
        return Err(Error::InvalidArgument {
            arg: "numtaps",
            reason: "Number of taps must be > 0".to_string(),
        });
    }

    if freq.len() != gain.len() {
        return Err(Error::InvalidArgument {
            arg: "freq/gain",
            reason: "freq and gain must have same length".to_string(),
        });
    }

    if freq.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "freq",
            reason: "freq must not be empty".to_string(),
        });
    }

    // Validate freq is monotonically increasing and in [0, 1]
    for i in 0..freq.len() {
        if freq[i] < 0.0 || freq[i] > 1.0 {
            return Err(Error::InvalidArgument {
                arg: "freq",
                reason: "freq values must be in [0, 1]".to_string(),
            });
        }
        if i > 0 && freq[i] <= freq[i - 1] {
            return Err(Error::InvalidArgument {
                arg: "freq",
                reason: "freq must be monotonically increasing".to_string(),
            });
        }
    }

    // irfft requires power-of-2 size
    let fft_size = numtaps.next_power_of_two();
    let nfreqs = fft_size / 2 + 1;

    // Interpolate gain to FFT grid
    let mut interp_gain = vec![0.0f64; nfreqs];
    for i in 0..nfreqs {
        let f = i as f64 / fft_size as f64;

        // Linear interpolation
        let mut g = 0.0;
        for j in 0..freq.len() - 1 {
            if f >= freq[j] && f <= freq[j + 1] {
                let t = (f - freq[j]) / (freq[j + 1] - freq[j]);
                g = gain[j] * (1.0 - t) + gain[j + 1] * t;
                break;
            }
        }
        if f <= freq[0] {
            g = gain[0];
        } else if f >= freq[freq.len() - 1] {
            g = gain[gain.len() - 1];
        }

        interp_gain[i] = g;
    }

    // Build frequency response
    let mut freq_resp_re = Vec::with_capacity(nfreqs);
    let mut freq_resp_im = Vec::with_capacity(nfreqs);

    if antisymmetric {
        for i in 0..nfreqs {
            freq_resp_re.push(0.0);
            freq_resp_im.push(interp_gain[i]);
        }
    } else {
        for i in 0..nfreqs {
            freq_resp_re.push(interp_gain[i]);
            freq_resp_im.push(0.0);
        }
    }

    // Apply linear phase
    let alpha = (numtaps - 1) as f64 / 2.0;
    for i in 0..nfreqs {
        let omega = PI * i as f64 / (fft_size as f64 / 2.0);
        let phase = -omega * alpha;
        let cos_p = phase.cos();
        let sin_p = phase.sin();
        let re = freq_resp_re[i];
        let im = freq_resp_im[i];
        freq_resp_re[i] = re * cos_p - im * sin_p;
        freq_resp_im[i] = re * sin_p + im * cos_p;
    }

    // Inverse FFT to get impulse response
    let freq_tensor = create_complex_tensor(&freq_resp_re, &freq_resp_im, device)?;
    let h_padded = client.irfft(
        &freq_tensor,
        Some(fft_size),
        numr::algorithm::fft::FftNormalization::Backward,
    )?;

    // Truncate to requested numtaps
    let h = h_padded.narrow(0, 0, numtaps)?;

    // Apply window
    let win = generate_window(client, numtaps, &window, DType::F32, device)?;
    client.mul(&h, &win)
}

/// Convert linear-phase FIR to minimum-phase (CPU-only).
fn minimum_phase_cpu(client: &CpuClient, h: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let n = h.shape()[0];
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "h",
            reason: "Filter must not be empty".to_string(),
        });
    }

    // Compute FFT
    let h_fft = client.rfft(h, numr::algorithm::fft::FftNormalization::None)?;

    // Get magnitude
    let re = client.real(&h_fft)?;
    let im = client.imag(&h_fft)?;
    let re_sq = client.mul(&re, &re)?;
    let im_sq = client.mul(&im, &im)?;
    let mag_sq = client.add(&re_sq, &im_sq)?;

    // Add small epsilon for numerical stability
    let eps = 1e-10;
    let mag_sq_safe = client.add_scalar(&mag_sq, eps)?;

    // Log magnitude
    let log_mag = client.log(&mag_sq_safe)?;
    let _half_log_mag = client.mul_scalar(&log_mag, 0.5)?;

    // The minimum phase filter has half the length
    let m = (n + 1) / 2;

    // For now, return a windowed version of the first half
    // (Complete implementation would use cepstrum method)
    let h_data: Vec<f64> = h.to_vec();
    let h_min: Vec<f64> = h_data[..m].to_vec();

    let device = h.device();
    Tensor::from_slice(&h_min, &[m], device)
}

// ============================================================================
// Helper functions
// ============================================================================

fn validate_fir_cutoff(cutoff: &[f64], filter_type: FilterType) -> Result<()> {
    match filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            if cutoff.len() != 1 {
                return Err(Error::InvalidArgument {
                    arg: "cutoff",
                    reason: format!("{:?} requires single cutoff frequency", filter_type),
                });
            }
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            if cutoff.len() != 2 {
                return Err(Error::InvalidArgument {
                    arg: "cutoff",
                    reason: format!("{:?} requires two cutoff frequencies", filter_type),
                });
            }
            if cutoff[0] >= cutoff[1] {
                return Err(Error::InvalidArgument {
                    arg: "cutoff",
                    reason: "Low cutoff must be less than high cutoff".to_string(),
                });
            }
        }
    }

    for &c in cutoff {
        if c <= 0.0 || c >= 1.0 {
            return Err(Error::InvalidArgument {
                arg: "cutoff",
                reason: "Cutoff frequencies must be in (0, 1)".to_string(),
            });
        }
    }

    Ok(())
}

fn generate_window(
    client: &CpuClient,
    size: usize,
    window: &FirWindow,
    dtype: DType,
    device: &CpuDevice,
) -> Result<Tensor<CpuRuntime>> {
    match window {
        FirWindow::Rectangular => Ok(Tensor::ones(&[size], dtype, device)?),
        FirWindow::Hann => client.hann_window(size, dtype, device),
        FirWindow::Hamming => client.hamming_window(size, dtype, device),
        FirWindow::Blackman => client.blackman_window(size, dtype, device),
        FirWindow::Kaiser(beta) => client.kaiser_window(size, *beta, dtype, device),
        FirWindow::Custom(coeffs) => {
            if coeffs.len() != size {
                return Err(Error::InvalidArgument {
                    arg: "window",
                    reason: format!(
                        "Custom window size {} doesn't match numtaps {}",
                        coeffs.len(),
                        size
                    ),
                });
            }
            Ok(Tensor::from_slice(coeffs, &[size], device)?)
        }
    }
}

fn compute_gain(h: &Tensor<CpuRuntime>, cutoff: &[f64], filter_type: FilterType) -> Result<f64> {
    let h_data: Vec<f64> = h.to_vec();

    // Compute gain at passband center frequency
    let freq = match filter_type {
        FilterType::Lowpass => 0.0,
        FilterType::Highpass => 1.0,
        FilterType::Bandpass => (cutoff[0] + cutoff[1]) / 2.0,
        FilterType::Bandstop => 0.0,
    };

    // H(ω) = Σ h[n] * e^(-jωn)
    let omega = PI * freq;
    let mut re = 0.0;
    let mut im = 0.0;

    for (i, &coeff) in h_data.iter().enumerate() {
        let angle = omega * i as f64;
        re += coeff * angle.cos();
        im -= coeff * angle.sin();
    }

    Ok((re * re + im * im).sqrt())
}

fn create_complex_tensor(re: &[f64], im: &[f64], device: &CpuDevice) -> Result<Tensor<CpuRuntime>> {
    use numr::dtype::Complex64;

    let n = re.len();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(Complex64::new(re[i] as f32, im[i] as f32));
    }
    Tensor::from_slice(&data, &[n], device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_firwin_lowpass() {
        let (client, device) = setup();

        let h = client
            .firwin(
                51,
                &[0.2],
                FilterType::Lowpass,
                FirWindow::Hamming,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[51]);

        // Verify basic filter properties
        let h_data: Vec<f64> = h.to_vec();

        // Center value should be positive and largest
        let center = 25;
        assert!(h_data[center] > 0.0, "Center tap should be positive");

        // Taps should decay away from center
        assert!(h_data[center] > h_data[center - 1].abs());
        assert!(h_data[center] > h_data[center + 1].abs());

        // Sum of coefficients should be approximately 1 (DC gain)
        let sum: f64 = h_data.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "DC gain should be ~1, got {}", sum);
    }

    #[test]
    fn test_firwin_highpass() {
        let (client, device) = setup();

        let h = client
            .firwin(
                51,
                &[0.3],
                FilterType::Highpass,
                FirWindow::Hann,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[51]);
    }

    #[test]
    fn test_firwin_bandpass() {
        let (client, device) = setup();

        let h = client
            .firwin(
                101,
                &[0.2, 0.4],
                FilterType::Bandpass,
                FirWindow::Blackman,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[101]);
    }

    #[test]
    fn test_firwin_kaiser() {
        let (client, device) = setup();

        let h = client
            .firwin(
                65,
                &[0.25],
                FilterType::Lowpass,
                FirWindow::Kaiser(5.0),
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[65]);
    }

    #[test]
    fn test_firwin2() {
        let (client, device) = setup();

        // Design a filter with specific frequency response
        let freq = vec![0.0, 0.2, 0.3, 1.0];
        let gain = vec![1.0, 1.0, 0.0, 0.0];

        let h = client
            .firwin2(51, &freq, &gain, false, FirWindow::Hamming, &device)
            .unwrap();

        assert_eq!(h.shape(), &[51]);
    }
}
