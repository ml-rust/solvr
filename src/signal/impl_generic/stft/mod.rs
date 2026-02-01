//! Generic STFT/ISTFT implementations.
//!
//! Short-Time Fourier Transform and inverse for spectral analysis.

#![allow(clippy::too_many_arguments)]

mod istft;

pub use istft::istft_impl;

use super::helpers::complex_magnitude_pow_impl;
use super::padding::pad_1d_reflect_impl;
use crate::signal::{stft_num_frames, validate_signal_dtype, validate_stft_params};
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::{Complex64, Complex128, DType};
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of STFT.
pub fn stft_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    n_fft: usize,
    hop_length: Option<usize>,
    window: Option<&Tensor<R>>,
    center: bool,
    _normalized: bool,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + WindowFunctions<R> + TensorOps<R> + RuntimeClient<R>,
{
    let dtype = signal.dtype();
    validate_signal_dtype(dtype, "stft")?;

    let hop = hop_length.unwrap_or(n_fft / 4);
    validate_stft_params(n_fft, hop, "stft")?;

    let signal_contig = signal.contiguous();
    let ndim = signal_contig.ndim();

    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "signal",
            reason: "stft requires at least 1D signal".to_string(),
        });
    }

    let signal_len = signal_contig.shape()[ndim - 1];

    // Get or create window
    let default_window;
    let win = if let Some(w) = window {
        if w.shape() != [n_fft] {
            return Err(Error::InvalidArgument {
                arg: "window",
                reason: format!("window must have shape [{n_fft}], got {:?}", w.shape()),
            });
        }
        w
    } else {
        default_window = client.hann_window(n_fft, dtype, client.device())?;
        &default_window
    };

    // Calculate number of frames
    let n_frames = stft_num_frames(signal_len, n_fft, hop, center);

    if n_frames == 0 {
        return Err(Error::InvalidArgument {
            arg: "signal",
            reason: format!("signal too short for STFT: length={signal_len}, n_fft={n_fft}"),
        });
    }

    // Pad signal if centering
    let padded_signal = if center {
        let pad_left = n_fft / 2;
        let pad_right = n_fft / 2;
        pad_1d_reflect_impl(client, &signal_contig, pad_left, pad_right)?
    } else {
        signal_contig.clone()
    };

    let batch_size: usize = if ndim > 1 {
        signal_contig.shape()[..ndim - 1].iter().product()
    } else {
        1
    };

    let freq_bins = n_fft / 2 + 1;
    let padded_len = padded_signal.shape()[padded_signal.ndim() - 1];

    // Output shape: [..., n_frames, freq_bins]
    let mut out_shape: Vec<usize> = signal_contig.shape()[..ndim - 1].to_vec();
    out_shape.push(n_frames);
    out_shape.push(freq_bins);

    // Process using generic to_vec/from_slice pattern
    // Note: normalization is handled inside the FFT call
    match dtype {
        DType::F32 => stft_process_f32(
            client,
            &padded_signal,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            padded_len,
        ),
        DType::F64 => stft_process_f64(
            client,
            &padded_signal,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            padded_len,
        ),
        _ => Err(Error::UnsupportedDType { dtype, op: "stft" }),
    }
}

fn stft_process_f32<R, C>(
    client: &C,
    signal: &Tensor<R>,
    window: &Tensor<R>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    signal_len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + RuntimeClient<R>,
{
    let signal_data: Vec<f32> = signal.to_vec();
    let window_data: Vec<f32> = window.to_vec();
    let mut output_data = vec![Complex64::new(0.0, 0.0); batch_size * n_frames * freq_bins];

    for b in 0..batch_size {
        let sig_offset = b * signal_len;

        for f in 0..n_frames {
            let frame_start = f * hop;

            // Extract and window frame
            let mut frame = vec![0.0f32; n_fft];
            for i in 0..n_fft {
                let sig_idx = frame_start + i;
                let sig_val = if sig_idx < signal_len {
                    signal_data[sig_offset + sig_idx]
                } else {
                    0.0
                };
                frame[i] = sig_val * window_data[i];
            }

            // Create tensor and compute rfft
            let frame_tensor = Tensor::<R>::from_slice(&frame, &[n_fft], client.device());
            let spectrum = client.rfft(&frame_tensor, FftNormalization::None)?;
            let spec_data: Vec<Complex64> = spectrum.to_vec();

            // Copy to output
            let out_offset = b * n_frames * freq_bins + f * freq_bins;
            output_data[out_offset..out_offset + freq_bins]
                .copy_from_slice(&spec_data[..freq_bins]);
        }
    }

    Ok(Tensor::<R>::from_slice(
        &output_data,
        out_shape,
        client.device(),
    ))
}

fn stft_process_f64<R, C>(
    client: &C,
    signal: &Tensor<R>,
    window: &Tensor<R>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    signal_len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + RuntimeClient<R>,
{
    let signal_data: Vec<f64> = signal.to_vec();
    let window_data: Vec<f64> = window.to_vec();
    let mut output_data = vec![Complex128::new(0.0, 0.0); batch_size * n_frames * freq_bins];

    for b in 0..batch_size {
        let sig_offset = b * signal_len;

        for f in 0..n_frames {
            let frame_start = f * hop;

            let mut frame = vec![0.0f64; n_fft];
            for i in 0..n_fft {
                let sig_idx = frame_start + i;
                let sig_val = if sig_idx < signal_len {
                    signal_data[sig_offset + sig_idx]
                } else {
                    0.0
                };
                frame[i] = sig_val * window_data[i];
            }

            let frame_tensor = Tensor::<R>::from_slice(&frame, &[n_fft], client.device());
            let spectrum = client.rfft(&frame_tensor, FftNormalization::None)?;
            let spec_data: Vec<Complex128> = spectrum.to_vec();

            let out_offset = b * n_frames * freq_bins + f * freq_bins;
            output_data[out_offset..out_offset + freq_bins]
                .copy_from_slice(&spec_data[..freq_bins]);
        }
    }

    Ok(Tensor::<R>::from_slice(
        &output_data,
        out_shape,
        client.device(),
    ))
}

/// Generic implementation of spectrogram.
pub fn spectrogram_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    n_fft: usize,
    hop_length: Option<usize>,
    window: Option<&Tensor<R>>,
    power: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + WindowFunctions<R> + TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let stft_result = stft_impl(client, signal, n_fft, hop_length, window, true, false)?;
    let dtype = signal.dtype();
    complex_magnitude_pow_impl(client, &stft_result, power, dtype)
}
