//! Shared STFT/ISTFT core algorithms.
//!
//! This module contains the generic STFT and ISTFT implementations that work on
//! slices and vectors. All backends (CPU, CUDA, WebGPU) use these core algorithms
//! to ensure numerical consistency.
//!
//! # Design Rationale
//!
//! STFT/ISTFT are inherently sequential algorithms (frame-by-frame FFT). While the
//! FFT itself can be GPU-accelerated, the frame extraction and overlap-add are
//! typically done on CPU. By extracting the common logic here, we:
//!
//! 1. Eliminate code duplication across backends (~650 lines reduced)
//! 2. Ensure identical numerical behavior across all backends
//! 3. Make the algorithm easier to test and maintain

use numr::dtype::{Complex128, Complex64};

/// Extract and window a single frame from a signal.
///
/// # Arguments
///
/// * `signal` - The input signal slice
/// * `window` - The window function slice
/// * `frame_start` - Starting index of the frame in the signal
/// * `frame_out` - Output buffer for the windowed frame (must be n_fft length)
///
/// # Safety
///
/// This function is safe - it performs bounds checking.
#[inline]
pub fn extract_windowed_frame_f32(
    signal: &[f32],
    window: &[f32],
    frame_start: usize,
    frame_out: &mut [f32],
) {
    let n_fft = frame_out.len();
    let signal_len = signal.len();

    for i in 0..n_fft {
        let sig_idx = frame_start + i;
        let sig_val = if sig_idx < signal_len {
            signal[sig_idx]
        } else {
            0.0
        };
        frame_out[i] = sig_val * window[i];
    }
}

/// Extract and window a single frame from a signal (f64 version).
#[inline]
pub fn extract_windowed_frame_f64(
    signal: &[f64],
    window: &[f64],
    frame_start: usize,
    frame_out: &mut [f64],
) {
    let n_fft = frame_out.len();
    let signal_len = signal.len();

    for i in 0..n_fft {
        let sig_idx = frame_start + i;
        let sig_val = if sig_idx < signal_len {
            signal[sig_idx]
        } else {
            0.0
        };
        frame_out[i] = sig_val * window[i];
    }
}

/// ISTFT overlap-add reconstruction for a single batch.
///
/// # Arguments
///
/// * `frames` - Iterator over (frame_index, frame_data) tuples
/// * `window` - The synthesis window
/// * `reconstruction` - Output buffer (full_len)
/// * `window_sum` - Window sum buffer for normalization (full_len)
/// * `n_fft` - FFT size
/// * `hop` - Hop length
///
/// After all frames are added, normalize by:
/// `output[i] = reconstruction[i] / max(window_sum[i], epsilon)`
pub fn overlap_add_f32<'a>(
    frames: impl Iterator<Item = (usize, &'a [f32])>,
    window: &[f32],
    reconstruction: &mut [f32],
    window_sum: &mut [f32],
    n_fft: usize,
    hop: usize,
) {
    let full_len = reconstruction.len();

    for (frame_idx, frame_data) in frames {
        let frame_start = frame_idx * hop;

        for i in 0..n_fft {
            let out_idx = frame_start + i;
            if out_idx < full_len {
                let win_val = window[i];
                reconstruction[out_idx] += frame_data[i] * win_val;
                window_sum[out_idx] += win_val * win_val;
            }
        }
    }
}

/// ISTFT overlap-add reconstruction for a single batch (f64 version).
pub fn overlap_add_f64<'a>(
    frames: impl Iterator<Item = (usize, &'a [f64])>,
    window: &[f64],
    reconstruction: &mut [f64],
    window_sum: &mut [f64],
    n_fft: usize,
    hop: usize,
) {
    let full_len = reconstruction.len();

    for (frame_idx, frame_data) in frames {
        let frame_start = frame_idx * hop;

        for i in 0..n_fft {
            let out_idx = frame_start + i;
            if out_idx < full_len {
                let win_val = window[i];
                reconstruction[out_idx] += frame_data[i] * win_val;
                window_sum[out_idx] += win_val * win_val;
            }
        }
    }
}

/// Normalize reconstruction by window sum and copy to output with optional centering.
///
/// # Arguments
///
/// * `reconstruction` - The overlap-add result
/// * `window_sum` - The window sum for normalization
/// * `output` - The final output buffer
/// * `pad_left` - Left padding to skip (for centered STFT)
/// * `epsilon` - Small value to avoid division by zero (typically 1e-8)
pub fn normalize_and_copy_f32(
    reconstruction: &[f32],
    window_sum: &[f32],
    output: &mut [f32],
    pad_left: usize,
    epsilon: f32,
) {
    let full_len = reconstruction.len();

    for (i, out_val) in output.iter_mut().enumerate() {
        let src_idx = pad_left + i;
        if src_idx < full_len {
            let norm_factor = if window_sum[src_idx] > epsilon {
                window_sum[src_idx]
            } else {
                1.0
            };
            *out_val = reconstruction[src_idx] / norm_factor;
        }
    }
}

/// Normalize reconstruction by window sum and copy to output (f64 version).
pub fn normalize_and_copy_f64(
    reconstruction: &[f64],
    window_sum: &[f64],
    output: &mut [f64],
    pad_left: usize,
    epsilon: f64,
) {
    let full_len = reconstruction.len();

    for (i, out_val) in output.iter_mut().enumerate() {
        let src_idx = pad_left + i;
        if src_idx < full_len {
            let norm_factor = if window_sum[src_idx] > epsilon {
                window_sum[src_idx]
            } else {
                1.0
            };
            *out_val = reconstruction[src_idx] / norm_factor;
        }
    }
}

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
#[inline]
pub fn complex_mul_c64(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    )
}

/// Complex multiplication for Complex128.
#[inline]
pub fn complex_mul_c128(a: Complex128, b: Complex128) -> Complex128 {
    Complex128::new(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    )
}

/// Compute complex magnitude raised to a power.
///
/// |z|^power = (re^2 + im^2)^(power/2)
#[inline]
pub fn magnitude_pow_f32(c: Complex64, power: f64) -> f32 {
    let mag_sq = c.re * c.re + c.im * c.im;
    if power == 2.0 {
        mag_sq
    } else if power == 1.0 {
        mag_sq.sqrt()
    } else {
        mag_sq.powf(power as f32 / 2.0)
    }
}

/// Compute complex magnitude raised to a power (f64 version).
#[inline]
pub fn magnitude_pow_f64(c: Complex128, power: f64) -> f64 {
    let mag_sq = c.re * c.re + c.im * c.im;
    if power == 2.0 {
        mag_sq
    } else if power == 1.0 {
        mag_sq.sqrt()
    } else {
        mag_sq.powf(power / 2.0)
    }
}

/// Reverse a 1D slice in-place.
pub fn reverse_1d_into<T: Copy>(src: &[T], dst: &mut [T]) {
    let len = src.len();
    assert_eq!(dst.len(), len);
    for i in 0..len {
        dst[i] = src[len - 1 - i];
    }
}

/// Reverse a 2D array (flip both dimensions).
pub fn reverse_2d_into<T: Copy>(src: &[T], dst: &mut [T], h: usize, w: usize) {
    assert_eq!(src.len(), h * w);
    assert_eq!(dst.len(), h * w);
    for i in 0..h {
        for j in 0..w {
            dst[i * w + j] = src[(h - 1 - i) * w + (w - 1 - j)];
        }
    }
}

/// Reflect padding for 1D signal.
///
/// Pads the signal by reflecting at the boundaries.
///
/// # Arguments
///
/// * `src` - Source signal of length `current_len`
/// * `dst` - Output buffer of length `current_len + pad_left + pad_right`
/// * `pad_left` - Number of elements to pad on the left
/// * `pad_right` - Number of elements to pad on the right
pub fn reflect_pad_1d<T: Copy>(
    src: &[T],
    dst: &mut [T],
    pad_left: usize,
    _pad_right: usize,
) {
    let current_len = src.len();
    let target_len = dst.len();

    // Left padding (reflected)
    for (i, dst_val) in dst.iter_mut().take(pad_left).enumerate() {
        let reflect_idx = (pad_left - i).min(current_len - 1);
        *dst_val = src[reflect_idx];
    }

    // Original data
    dst[pad_left..pad_left + current_len].copy_from_slice(src);

    // Right padding (reflected)
    for i in 0..(target_len - pad_left - current_len) {
        let idx = if current_len <= 1 {
            0
        } else {
            let period = current_len - 1;
            let pos = i % (2 * period);
            if pos < period {
                current_len - 2 - pos
            } else {
                pos - period + 1
            }
        };
        dst[pad_left + current_len + i] = src[idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_windowed_frame() {
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let window = [0.5f32, 1.0, 0.5];
        let mut frame = [0.0f32; 3];

        extract_windowed_frame_f32(&signal, &window, 1, &mut frame);

        assert_eq!(frame[0], 2.0 * 0.5); // signal[1] * window[0]
        assert_eq!(frame[1], 3.0 * 1.0); // signal[2] * window[1]
        assert_eq!(frame[2], 4.0 * 0.5); // signal[3] * window[2]
    }

    #[test]
    fn test_extract_windowed_frame_boundary() {
        let signal = [1.0f32, 2.0, 3.0];
        let window = [1.0f32; 4];
        let mut frame = [0.0f32; 4];

        // Frame extends beyond signal
        extract_windowed_frame_f32(&signal, &window, 1, &mut frame);

        assert_eq!(frame[0], 2.0);
        assert_eq!(frame[1], 3.0);
        assert_eq!(frame[2], 0.0); // Zero-padded
        assert_eq!(frame[3], 0.0); // Zero-padded
    }

    #[test]
    fn test_complex_mul() {
        let a = Complex64::new(3.0, 2.0);
        let b = Complex64::new(1.0, 4.0);
        let result = complex_mul_c64(a, b);

        // (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2 = 3 + 14i - 8 = -5 + 14i
        assert!((result.re - (-5.0)).abs() < 1e-10);
        assert!((result.im - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_pow() {
        let c = Complex64::new(3.0, 4.0);

        // |3 + 4i| = 5, |z|^2 = 25
        assert!((magnitude_pow_f32(c, 2.0) - 25.0).abs() < 1e-10);
        assert!((magnitude_pow_f32(c, 1.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_1d() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        reverse_1d_into(&src, &mut dst);
        assert_eq!(dst, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reverse_2d() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0f32; 6];
        reverse_2d_into(&src, &mut dst, 2, 3);
        // Original: [[1,2,3], [4,5,6]]
        // Reversed: [[6,5,4], [3,2,1]]
        assert_eq!(dst, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
