//! Simple periodogram computation.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use crate::signal::impl_generic::helpers::{DetrendMode, detrend_tensor_impl, power_spectrum_impl};
use crate::signal::impl_generic::spectral::helpers::generate_window;
use crate::signal::traits::spectral::{Detrend, PeriodogramParams, PeriodogramResult, PsdScaling};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute periodogram using numr FFT operations.
///
/// Uses tensor operations throughout - no CPU transfers.
pub fn periodogram_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    params: PeriodogramParams<R>,
) -> Result<PeriodogramResult<R>>
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

    // Determine nfft (power of 2 for FFT)
    let nfft_requested = params.nfft.unwrap_or(n).max(n);
    let nfft = nfft_requested.next_power_of_two();

    // Generate window tensor
    let window = generate_window(&params.window, n, &params.device);

    // Compute window sum of squares for normalization
    let win_sq = client.mul(&window, &window)?;
    let win_sum_sq_tensor = client.sum(&win_sq, &[0], false)?;
    let win_sum_sq: f64 = win_sum_sq_tensor.item()?;

    // Apply detrending
    let detrend_mode = match params.detrend {
        Detrend::None => DetrendMode::None,
        Detrend::Constant => DetrendMode::Constant,
        Detrend::Linear => DetrendMode::Linear,
    };
    let x_detrended = detrend_tensor_impl(client, x, detrend_mode)?;

    // Apply window
    let x_windowed = client.mul(&x_detrended, &window)?;

    // Pad to nfft if necessary
    let x_padded = if nfft > n {
        let pad_amount = nfft - n;
        client.pad(&x_windowed, &[0, pad_amount], 0.0)?
    } else {
        x_windowed
    };

    // Compute FFT via numr
    let fft_result = client.rfft(&x_padded, FftNormalization::None)?;

    // Compute power spectrum: |FFT|²
    let power = power_spectrum_impl(client, &fft_result)?;

    // Determine output frequencies
    let n_freqs = nfft / 2 + 1;

    // Apply scaling
    let scale = match params.scaling {
        PsdScaling::Density => 1.0 / (params.fs * win_sum_sq),
        PsdScaling::Spectrum => 1.0 / win_sum_sq,
    };

    let psd_scaled = client.mul_scalar(&power, scale)?;

    // For one-sided, double the non-DC, non-Nyquist components
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

    Ok(PeriodogramResult {
        freqs,
        psd: psd_final,
    })
}
