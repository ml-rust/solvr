//! Welch's method for power spectral density estimation.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use crate::signal::impl_generic::helpers::{
    DetrendMode, detrend_tensor_impl, extract_segments_impl, power_spectrum_impl,
};
use crate::signal::impl_generic::spectral::helpers::generate_window;
use crate::signal::traits::spectral::{Detrend, PsdScaling, WelchParams, WelchResult};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute power spectral density using Welch's method.
///
/// Uses batched FFT operations - backend-optimized via numr.
pub fn welch_impl<R, C>(client: &C, x: &Tensor<R>, params: WelchParams<R>) -> Result<WelchResult<R>>
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
    let n = x.shape()[0];

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    // Determine segment parameters
    let nperseg = params.nperseg.unwrap_or(256.min(n));
    let noverlap = params.noverlap.unwrap_or(nperseg / 2);
    let nfft = params.nfft.unwrap_or(nperseg).max(nperseg);

    // FFT requires power of 2
    let nfft = nfft.next_power_of_two();

    if nperseg > n {
        return Err(Error::InvalidArgument {
            arg: "nperseg",
            reason: format!(
                "nperseg ({}) cannot be greater than signal length ({})",
                nperseg, n
            ),
        });
    }

    if noverlap >= nperseg {
        return Err(Error::InvalidArgument {
            arg: "noverlap",
            reason: "noverlap must be less than nperseg".to_string(),
        });
    }

    // Generate window tensor
    let window = generate_window(&params.window, nperseg, &params.device);

    // Compute window sum of squares for normalization
    let win_sq = client.mul(&window, &window)?;
    let win_sum_sq_tensor = client.sum(&win_sq, &[0], false)?;
    let win_sum_sq: f64 = win_sum_sq_tensor.item()?;

    // Extract overlapping segments as 2D tensor [num_segments, nperseg]
    let segments = extract_segments_impl(client, x, nperseg, noverlap)?;
    let num_segments = segments.shape()[0];

    // Apply detrending to each segment
    let detrend_mode = match params.detrend {
        Detrend::None => DetrendMode::None,
        Detrend::Constant => DetrendMode::Constant,
        Detrend::Linear => DetrendMode::Linear,
    };
    let segments_detrended = detrend_tensor_impl(client, &segments, detrend_mode)?;

    // Apply window to each segment (broadcast window across segments)
    let window_broadcast = window.reshape(&[1, nperseg])?;
    let segments_windowed = client.mul(&segments_detrended, &window_broadcast)?;

    // Pad to nfft if necessary
    let segments_padded = if nfft > nperseg {
        let pad_amount = nfft - nperseg;
        // Pad on the right (last dimension)
        client.pad(&segments_windowed, &[0, pad_amount], 0.0)?
    } else {
        segments_windowed
    };

    // Compute batch FFT via numr
    // rfft on last dimension gives [num_segments, nfft/2 + 1] complex output
    let fft_result = client.rfft(&segments_padded, FftNormalization::None)?;

    // Compute power spectrum: |FFT|²
    let power = power_spectrum_impl(client, &fft_result)?;

    // Average across segments (dim 0)
    let psd_sum = client.sum(&power, &[0], false)?;
    let psd_avg = client.div_scalar(&psd_sum, num_segments as f64)?;

    // Determine output frequencies
    let n_freqs = nfft / 2 + 1;

    // Apply scaling
    let scale = match params.scaling {
        PsdScaling::Density => 1.0 / (params.fs * win_sum_sq),
        PsdScaling::Spectrum => 1.0 / win_sum_sq,
    };

    let psd_scaled = client.mul_scalar(&psd_avg, scale)?;

    // For one-sided, double the non-DC, non-Nyquist components
    // This is done via tensor operations
    let psd_final = if params.onesided && n_freqs > 2 {
        // Create scaling mask: [1, 2, 2, ..., 2, 1] for DC and Nyquist = 1, others = 2
        let mut scale_factors = vec![2.0f64; n_freqs];
        scale_factors[0] = 1.0; // DC
        if n_freqs > 1 {
            scale_factors[n_freqs - 1] = 1.0; // Nyquist
        }
        let scale_tensor = Tensor::from_slice(&scale_factors, &[n_freqs], &params.device);
        client.mul(&psd_scaled, &scale_tensor)?
    } else {
        psd_scaled
    };

    // Generate frequency bins using rfftfreq
    let freqs = client.rfftfreq(nfft, 1.0 / params.fs, psd_final.dtype(), &params.device)?;

    Ok(WelchResult {
        freqs,
        psd: psd_final,
    })
}
