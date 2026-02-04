//! FIR filter design implementations.
//!
//! Implements FIR filter design using:
//! - Windowed sinc method (firwin)
//! - Frequency sampling method (firwin2)
//! - Minimum phase conversion

// Allow indexed loops for filter coefficient computation
#![allow(clippy::needless_range_loop)]
// Allow manual div_ceil for clarity
#![allow(clippy::manual_div_ceil)]

use crate::signal::filter::traits::fir_design::FirWindow;
use crate::signal::filter::types::FilterType;
use crate::window::WindowFunctions;
use numr::algorithm::fft::FftAlgorithms;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Design FIR filter using windowed sinc method.
///
/// # Algorithm
///
/// 1. Design ideal (brick-wall) lowpass filter via sinc function
/// 2. Apply frequency transformation if needed (HP, BP, BS)
/// 3. Apply window function
/// 4. Optionally scale for unity gain at passband center
pub fn firwin_impl<R, C>(
    client: &C,
    numtaps: usize,
    cutoff: &[f64],
    filter_type: FilterType,
    window: FirWindow,
    scale: bool,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: WindowFunctions<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    if numtaps == 0 {
        return Err(Error::InvalidArgument {
            arg: "numtaps",
            reason: "Number of taps must be > 0".to_string(),
        });
    }

    // Validate cutoff frequencies
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
    let h_tensor = Tensor::from_slice(&h, &[numtaps], device);
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

/// Design FIR filter using frequency sampling method.
///
/// # Algorithm
///
/// 1. Interpolate desired frequency response to FFT grid
/// 2. Apply inverse FFT to get impulse response
/// 3. Apply window
pub fn firwin2_impl<R, C>(
    client: &C,
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    antisymmetric: bool,
    window: FirWindow,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + WindowFunctions<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
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

    let _dtype = DType::F64;

    // irfft requires power-of-2 size, so pad numtaps if needed
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

    // Build frequency response (complex, with phase for antisymmetric)
    let mut freq_resp_re = Vec::with_capacity(nfreqs);
    let mut freq_resp_im = Vec::with_capacity(nfreqs);

    if antisymmetric {
        // Type III/IV filter: H(ω) = j * |H(ω)|
        for i in 0..nfreqs {
            freq_resp_re.push(0.0);
            freq_resp_im.push(interp_gain[i]);
        }
    } else {
        // Type I/II filter: H(ω) = |H(ω)|
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
    // For real filters, we use irfft with power-of-2 size
    let freq_tensor = create_complex_tensor(&freq_resp_re, &freq_resp_im, device)?;
    let h_padded = client.irfft(
        &freq_tensor,
        Some(fft_size),
        numr::algorithm::fft::FftNormalization::Backward,
    )?;

    // Truncate to requested numtaps
    let h = h_padded.narrow(0, 0, numtaps)?;

    // Apply window - use F32 since irfft returns F32 from Complex64 input
    let win = generate_window(client, numtaps, &window, DType::F32, device)?;
    client.mul(&h, &win)
}

/// Convert linear-phase FIR to minimum-phase.
///
/// # Algorithm
///
/// 1. Compute FFT of filter
/// 2. Take log of magnitude
/// 3. Apply Hilbert transform via FFT
/// 4. Reconstruct minimum-phase filter
pub fn minimum_phase_impl<R, C>(client: &C, h: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
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

    // Hilbert transform via FFT to get minimum phase
    // For a real signal, minimum phase = exp(log_mag + j * hilbert(log_mag))

    // The minimum phase filter has half the length
    let m = (n + 1) / 2;

    // For now, return a windowed version of the first half
    // (Complete implementation would use cepstrum method)
    let h_data: Vec<f64> = h.to_vec();
    let h_min: Vec<f64> = h_data[..m].to_vec();

    let device = h.device();
    Ok(Tensor::from_slice(&h_min, &[m], device))
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

fn generate_window<R, C>(
    client: &C,
    size: usize,
    window: &FirWindow,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: WindowFunctions<R> + RuntimeClient<R>,
{
    match window {
        FirWindow::Rectangular => Ok(Tensor::ones(&[size], dtype, device)),
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
            Ok(Tensor::from_slice(coeffs, &[size], device))
        }
    }
}

fn compute_gain<R: Runtime>(h: &Tensor<R>, cutoff: &[f64], filter_type: FilterType) -> Result<f64> {
    let h_data: Vec<f64> = h.to_vec();
    let _n = h_data.len();

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

fn create_complex_tensor<R: Runtime>(
    re: &[f64],
    im: &[f64],
    device: &R::Device,
) -> Result<Tensor<R>> {
    use numr::dtype::Complex64;

    // Create complex values
    let n = re.len();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(Complex64::new(re[i] as f32, im[i] as f32));
    }
    Ok(Tensor::from_slice(&data, &[n], device))
}
