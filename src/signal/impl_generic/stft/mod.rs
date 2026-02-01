//! Generic STFT/ISTFT implementations.
//!
//! Short-Time Fourier Transform and inverse for spectral analysis.
//! All computation stays on device - no GPU->CPU->GPU roundtrips.

#![allow(clippy::too_many_arguments)]

mod istft;

pub use istft::istft_impl;

use super::helpers::complex_magnitude_pow_impl;
use super::padding::pad_1d_reflect_impl;
use crate::signal::{stft_num_frames, validate_signal_dtype, validate_stft_params};
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of STFT.
///
/// All computation stays on device - no to_vec() calls in the algorithm loop.
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
    C: FftAlgorithms<R> + WindowFunctions<R> + TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
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

    // Process based on batch size
    if batch_size == 1 {
        stft_single(client, &padded_signal, win, n_fft, hop, n_frames, freq_bins)
    } else {
        stft_batched(
            client,
            &padded_signal,
            win,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
        )
    }
}

/// STFT for a single signal (no batch dimension).
///
/// All operations stay on device.
fn stft_single<R, C>(
    client: &C,
    signal: &Tensor<R>,
    window: &Tensor<R>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    freq_bins: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + TensorOps<R> + RuntimeClient<R>,
{
    let signal_len = signal.shape()[0];

    // Collect all frame spectra
    let mut frame_spectra: Vec<Tensor<R>> = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let frame_start = f * hop;

        // Extract frame using narrow (stays on device)
        // Handle case where frame might extend past signal end
        let available = signal_len.saturating_sub(frame_start);
        let frame_len = n_fft.min(available);

        let frame = if frame_len == n_fft && frame_start + n_fft <= signal_len {
            // Normal case: full frame available
            signal.narrow(0, frame_start, n_fft)?.contiguous()
        } else {
            // Edge case: need to pad with zeros
            // Extract what we can and pad the rest
            if frame_len > 0 {
                let partial = signal.narrow(0, frame_start, frame_len)?.contiguous();
                let pad_amount = n_fft - frame_len;
                client.pad(&partial, &[0, pad_amount], 0.0)?
            } else {
                // No signal left, create zeros
                Tensor::<R>::zeros(&[n_fft], signal.dtype(), client.device())
            }
        };

        // Apply window (stays on device)
        let windowed = client.mul(&frame, window)?;

        // RFFT to get spectrum (stays on device)
        let spectrum = client.rfft(&windowed, FftNormalization::None)?;

        // Reshape to [1, freq_bins] for stacking
        let spectrum_2d = spectrum.reshape(&[1, freq_bins])?;
        frame_spectra.push(spectrum_2d);
    }

    // Stack all frames along dimension 0 to get [n_frames, freq_bins]
    let refs: Vec<&Tensor<R>> = frame_spectra.iter().collect();
    client.cat(&refs, 0)
}

/// STFT for batched signals.
///
/// All operations stay on device.
fn stft_batched<R, C>(
    client: &C,
    signal: &Tensor<R>,
    window: &Tensor<R>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + TensorOps<R> + RuntimeClient<R>,
{
    // Reshape signal to [batch_size, signal_len]
    let signal_len = signal.numel() / batch_size;
    let signal_2d = signal.reshape(&[batch_size, signal_len])?;

    // Broadcast window to [1, n_fft] for batch multiplication
    let window_2d = window.reshape(&[1, n_fft])?;

    // Collect all frame spectra
    let mut frame_spectra: Vec<Tensor<R>> = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let frame_start = f * hop;

        // Extract frame for all batches: [batch_size, n_fft]
        let available = signal_len.saturating_sub(frame_start);
        let frame_len = n_fft.min(available);

        let frames = if frame_len == n_fft && frame_start + n_fft <= signal_len {
            // Normal case: full frame available
            signal_2d.narrow(1, frame_start, n_fft)?.contiguous()
        } else {
            // Edge case: need to pad with zeros
            if frame_len > 0 {
                let partial = signal_2d.narrow(1, frame_start, frame_len)?.contiguous();
                let pad_amount = n_fft - frame_len;
                client.pad(&partial, &[0, pad_amount], 0.0)?
            } else {
                Tensor::<R>::zeros(&[batch_size, n_fft], signal.dtype(), client.device())
            }
        };

        // Apply window (broadcasts [1, n_fft] to [batch_size, n_fft])
        let windowed = client.mul(&frames, &window_2d)?;

        // RFFT to get spectrum: [batch_size, freq_bins]
        let spectrum = client.rfft(&windowed, FftNormalization::None)?;

        // Reshape to [batch_size, 1, freq_bins] for stacking
        let spectrum_3d = spectrum.reshape(&[batch_size, 1, freq_bins])?;
        frame_spectra.push(spectrum_3d);
    }

    // Stack all frames along dimension 1 to get [batch_size, n_frames, freq_bins]
    let refs: Vec<&Tensor<R>> = frame_spectra.iter().collect();
    client.cat(&refs, 1)
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
