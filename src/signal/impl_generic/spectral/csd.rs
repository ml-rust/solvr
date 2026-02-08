//! Cross spectral density estimation using Welch's method.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use crate::signal::impl_generic::helpers::{
    DetrendMode, detrend_tensor_impl, extract_segments_impl,
};
use crate::signal::impl_generic::spectral::helpers::generate_window;
use crate::signal::traits::spectral::{CsdResult, Detrend, PsdScaling, WelchParams};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute cross spectral density using Welch's method.
///
/// Uses batched FFT operations - backend-optimized via numr.
pub fn csd_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    params: WelchParams<R>,
) -> Result<CsdResult<R>>
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

    // Extract overlapping segments as 2D tensors [num_segments, nperseg]
    let x_segments = extract_segments_impl(client, x, nperseg, noverlap)?;
    let y_segments = extract_segments_impl(client, y, nperseg, noverlap)?;
    let num_segments = x_segments.shape()[0];

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

    // Cross spectrum: conj(X) * Y
    let x_conj = client.conj(&x_fft)?;
    let pxy_complex = client.mul(&x_conj, &y_fft)?;

    // Average across segments (dim 0)
    let pxy_sum = client.sum(&pxy_complex, &[0], false)?;
    let pxy_avg = client.div_scalar(&pxy_sum, num_segments as f64)?;

    // Determine output frequencies
    let n_freqs = nfft / 2 + 1;

    // Apply scaling
    let scale = match params.scaling {
        PsdScaling::Density => 1.0 / (params.fs * win_sum_sq),
        PsdScaling::Spectrum => 1.0 / win_sum_sq,
    };

    let pxy_scaled = client.mul_scalar(&pxy_avg, scale)?;

    // Extract real and imaginary parts first (to avoid Complex * F64 tensor issues)
    let pxy_real_base = client.real(&pxy_scaled)?;
    let pxy_imag_base = client.imag(&pxy_scaled)?;

    // For one-sided, double the non-DC, non-Nyquist components
    let (pxy_real, pxy_imag) = if params.onesided && n_freqs > 2 {
        let mut scale_factors = vec![2.0f64; n_freqs];
        scale_factors[0] = 1.0; // DC
        if n_freqs > 1 {
            scale_factors[n_freqs - 1] = 1.0; // Nyquist
        }
        let scale_tensor = Tensor::from_slice(&scale_factors, &[n_freqs], &params.device);
        (
            client.mul(&pxy_real_base, &scale_tensor)?,
            client.mul(&pxy_imag_base, &scale_tensor)?,
        )
    } else {
        (pxy_real_base, pxy_imag_base)
    };

    // Generate frequency bins using rfftfreq
    let freqs = client.rfftfreq(nfft, 1.0 / params.fs, pxy_real.dtype(), &params.device)?;

    Ok(CsdResult {
        freqs,
        pxy_real,
        pxy_imag,
    })
}
