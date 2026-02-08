//! Magnitude squared coherence estimation.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use crate::signal::impl_generic::helpers::{
    DetrendMode, detrend_tensor_impl, extract_segments_impl, power_spectrum_impl,
};
use crate::signal::impl_generic::spectral::helpers::generate_window;
use crate::signal::traits::spectral::{CoherenceResult, Detrend, WelchParams};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute magnitude squared coherence.
///
/// Uses batched FFT operations - backend-optimized via numr.
///
/// Coherence is computed as: Cxy = |Pxy|² / (Pxx * Pyy)
/// where Pxx, Pyy are auto-spectra and Pxy is the cross-spectrum.
pub fn coherence_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    params: WelchParams<R>,
) -> Result<CoherenceResult<R>>
where
    R: Runtime,
    C: FftAlgorithms<R>
        + ComplexOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let nx = x.shape()[0];
    let ny = y.shape()[0];

    if nx != ny {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "x and y must have the same length".to_string(),
        });
    }

    let n = nx;

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signals cannot be empty".to_string(),
        });
    }

    // Determine segment parameters
    let nperseg = params.nperseg.unwrap_or(256.min(n));
    let noverlap = params.noverlap.unwrap_or(nperseg / 2);
    let nfft = params.nfft.unwrap_or(nperseg).max(nperseg);

    // FFT requires power of 2
    let nfft = nfft.next_power_of_two();

    if nperseg > n || noverlap >= nperseg {
        return Err(Error::InvalidArgument {
            arg: "nperseg",
            reason: "Invalid segment parameters".to_string(),
        });
    }

    // Generate window tensor
    let window = generate_window(&params.window, nperseg, &params.device);

    // Extract overlapping segments as 2D tensors [num_segments, nperseg]
    let x_segments = extract_segments_impl(client, x, nperseg, noverlap)?;
    let y_segments = extract_segments_impl(client, y, nperseg, noverlap)?;

    // Apply detrending to each segment
    let detrend_mode = match params.detrend {
        Detrend::None => DetrendMode::None,
        Detrend::Constant => DetrendMode::Constant,
        Detrend::Linear => DetrendMode::Linear,
    };
    let x_detrended = detrend_tensor_impl(client, &x_segments, detrend_mode)?;
    let y_detrended = detrend_tensor_impl(client, &y_segments, detrend_mode)?;

    // Apply window to each segment (broadcast window across segments)
    let window_broadcast = window.reshape(&[1, nperseg])?;
    let x_windowed = client.mul(&x_detrended, &window_broadcast)?;
    let y_windowed = client.mul(&y_detrended, &window_broadcast)?;

    // Pad to nfft if necessary
    let x_padded = if nfft > nperseg {
        let pad_amount = nfft - nperseg;
        client.pad(&x_windowed, &[0, pad_amount], 0.0)?
    } else {
        x_windowed
    };

    let y_padded = if nfft > nperseg {
        let pad_amount = nfft - nperseg;
        client.pad(&y_windowed, &[0, pad_amount], 0.0)?
    } else {
        y_windowed
    };

    // Compute batch FFT via numr
    let x_fft = client.rfft(&x_padded, FftNormalization::None)?;
    let y_fft = client.rfft(&y_padded, FftNormalization::None)?;

    // Compute Pxx = |X|² (power spectrum of x)
    let pxx = power_spectrum_impl(client, &x_fft)?;
    let pxx_sum = client.sum(&pxx, &[0], false)?;

    // Compute Pyy = |Y|² (power spectrum of y)
    let pyy = power_spectrum_impl(client, &y_fft)?;
    let pyy_sum = client.sum(&pyy, &[0], false)?;

    // Compute Pxy = conj(X) * Y (cross-spectrum)
    let x_conj = client.conj(&x_fft)?;
    let pxy_complex = client.mul(&x_conj, &y_fft)?;
    let pxy_sum = client.sum(&pxy_complex, &[0], false)?;

    // Compute |Pxy|² = real(Pxy)² + imag(Pxy)²
    let pxy_conj = client.conj(&pxy_sum)?;
    let pxy_mag_sq_complex = client.mul(&pxy_conj, &pxy_sum)?;
    let pxy_mag_sq = client.real(&pxy_mag_sq_complex)?;

    // Compute coherence: Cxy = |Pxy|² / (Pxx * Pyy)
    let pxx_pyy = client.mul(&pxx_sum, &pyy_sum)?;

    // Avoid division by zero - add small epsilon
    let epsilon = 1e-30;
    let pxx_pyy_safe = client.add_scalar(&pxx_pyy, epsilon)?;

    let cxy = client.div(&pxy_mag_sq, &pxx_pyy_safe)?;

    // Clamp to [0, 1] range
    let zeros = Tensor::zeros(cxy.shape(), cxy.dtype(), &params.device);
    let ones = Tensor::ones(cxy.shape(), cxy.dtype(), &params.device);
    let cxy_clamped = client.maximum(&zeros, &cxy)?;
    let cxy_final = client.minimum(&ones, &cxy_clamped)?;

    // Determine output frequencies
    let n_freqs = nfft / 2 + 1;

    // Generate frequency bins using rfftfreq
    let freqs = client.rfftfreq(
        n_freqs * 2 - 2,
        1.0 / params.fs,
        cxy_final.dtype(),
        &params.device,
    )?;

    // Narrow to actual frequency count (rfftfreq generates for full nfft)
    let freqs_final = freqs.narrow(0, 0, n_freqs)?;

    Ok(CoherenceResult {
        freqs: freqs_final,
        cxy: cxy_final,
    })
}
