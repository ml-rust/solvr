//! Generic ISTFT (Inverse Short-Time Fourier Transform) implementation.
//!
//! This implementation keeps all data on device using tensor operations.
//! No GPU->CPU->GPU roundtrips in the algorithm loop.

#![allow(clippy::too_many_arguments)]

use crate::signal::validate_stft_params;
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of ISTFT.
///
/// All computation stays on device - no to_vec() calls in the algorithm loop.
pub fn istft_impl<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    hop_length: Option<usize>,
    window: Option<&Tensor<R>>,
    center: bool,
    length: Option<usize>,
    normalized: bool,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + WindowFunctions<R> + TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let dtype = stft_matrix.dtype();

    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType { dtype, op: "istft" });
    }

    let real_dtype = match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => unreachable!(),
    };

    let stft_contig = stft_matrix.contiguous();
    let ndim = stft_contig.ndim();

    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "stft_matrix",
            reason: "istft requires at least 2D input [n_frames, freq_bins]".to_string(),
        });
    }

    let n_frames = stft_contig.shape()[ndim - 2];
    let freq_bins = stft_contig.shape()[ndim - 1];
    let n_fft = (freq_bins - 1) * 2;

    let hop = hop_length.unwrap_or(n_fft / 4);
    validate_stft_params(n_fft, hop, "istft")?;

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
        default_window = client.hann_window(n_fft, real_dtype, client.device())?;
        &default_window
    };

    let batch_size: usize = if ndim > 2 {
        stft_contig.shape()[..ndim - 2].iter().product()
    } else {
        1
    };

    // Calculate output length
    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };
    let output_len = if center {
        full_len - n_fft // Remove padding
    } else {
        full_len
    };
    let final_len = length.unwrap_or(output_len);

    let norm = if normalized {
        FftNormalization::Ortho
    } else {
        FftNormalization::Backward
    };

    // Compute window squared for normalization (stays on device)
    let window_sq = client.mul(win, win)?;

    // Process based on batch size
    if batch_size == 1 {
        // Single signal case - simpler code path
        istft_single(
            client,
            &stft_contig,
            win,
            &window_sq,
            real_dtype,
            n_fft,
            hop,
            n_frames,
            freq_bins,
            full_len,
            pad_left,
            final_len,
            norm,
        )
    } else {
        // Batched case
        istft_batched(
            client,
            &stft_contig,
            win,
            &window_sq,
            real_dtype,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            full_len,
            pad_left,
            final_len,
            norm,
        )
    }
}

/// ISTFT for a single signal (no batch dimension).
///
/// All operations stay on device.
fn istft_single<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    window: &Tensor<R>,
    window_sq: &Tensor<R>,
    real_dtype: DType,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    freq_bins: usize,
    full_len: usize,
    pad_left: usize,
    final_len: usize,
    norm: FftNormalization,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();

    // Initialize output and window sum tensors (on device)
    let mut output = Tensor::<R>::zeros(&[full_len], real_dtype, device);
    let mut window_sum = Tensor::<R>::zeros(&[full_len], real_dtype, device);

    // Process each frame - all operations stay on device
    for f in 0..n_frames {
        // Extract frame spectrum using narrow (stays on device)
        // stft_matrix shape: [n_frames, freq_bins]
        // narrow returns a view; make contiguous before reshape
        let spectrum = stft_matrix
            .narrow(0, f, 1)?
            .contiguous()
            .reshape(&[freq_bins])?;

        // IRFFT to get time-domain frame (stays on device)
        let frame = client.irfft(&spectrum, Some(n_fft), norm)?;

        // Apply window (stays on device)
        let windowed_frame = client.mul(&frame, window)?;

        // Pad frame to full length at the correct position
        let frame_start = f * hop;
        let right_pad = full_len.saturating_sub(frame_start + n_fft);

        // Pad: [frame_start zeros] [frame] [right_pad zeros]
        let padded_frame = client.pad(&windowed_frame, &[frame_start, right_pad], 0.0)?;

        // Accumulate into output (stays on device)
        output = client.add(&output, &padded_frame)?;

        // Same for window normalization
        let padded_window_sq = client.pad(window_sq, &[frame_start, right_pad], 0.0)?;
        window_sum = client.add(&window_sum, &padded_window_sq)?;
    }

    // Avoid division by zero: clamp window_sum to minimum value
    let eps = Tensor::<R>::full_scalar(&[full_len], real_dtype, 1e-8, device);
    let safe_window_sum = client.maximum(&window_sum, &eps)?;

    // Normalize by window sum (stays on device)
    let normalized_output = client.div(&output, &safe_window_sum)?;

    // Extract the final output region
    if pad_left == 0 && final_len == full_len {
        Ok(normalized_output)
    } else {
        // Extract [pad_left : pad_left + final_len]
        let extracted =
            normalized_output.narrow(0, pad_left, final_len.min(full_len - pad_left))?;
        Ok(extracted.contiguous())
    }
}

/// ISTFT for batched signals.
///
/// All operations stay on device.
fn istft_batched<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    window: &Tensor<R>,
    window_sq: &Tensor<R>,
    real_dtype: DType,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    full_len: usize,
    pad_left: usize,
    final_len: usize,
    norm: FftNormalization,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();

    // Initialize output and window sum tensors (on device)
    // Shape: [batch_size, full_len]
    let mut output = Tensor::<R>::zeros(&[batch_size, full_len], real_dtype, device);
    let mut window_sum = Tensor::<R>::zeros(&[batch_size, full_len], real_dtype, device);

    // Reshape stft to [batch_size, n_frames, freq_bins]
    let stft_batched = stft_matrix.reshape(&[batch_size, n_frames, freq_bins])?;

    // Process each frame - all operations stay on device
    for f in 0..n_frames {
        // Extract frame spectrum for all batches: [batch_size, freq_bins]
        // narrow returns a view; make contiguous before reshape
        let spectrum = stft_batched
            .narrow(1, f, 1)?
            .contiguous()
            .reshape(&[batch_size, freq_bins])?;

        // IRFFT to get time-domain frames: [batch_size, n_fft]
        let frames = client.irfft(&spectrum, Some(n_fft), norm)?;

        // Broadcast window to batch: [1, n_fft] -> broadcasts with [batch_size, n_fft]
        let window_broadcast = window.reshape(&[1, n_fft])?;
        let window_sq_broadcast = window_sq.reshape(&[1, n_fft])?;

        // Apply window (stays on device)
        let windowed_frames = client.mul(&frames, &window_broadcast)?;

        // Pad frames to full length at the correct position
        let frame_start = f * hop;
        let right_pad = full_len.saturating_sub(frame_start + n_fft);

        // Pad along last dimension: [batch_size, full_len]
        let padded_frames = client.pad(&windowed_frames, &[frame_start, right_pad], 0.0)?;

        // Accumulate into output (stays on device)
        output = client.add(&output, &padded_frames)?;

        // Same for window normalization
        let padded_window_sq = client.pad(&window_sq_broadcast, &[frame_start, right_pad], 0.0)?;
        // Broadcast padded_window_sq to batch
        window_sum = client.add(&window_sum, &padded_window_sq)?;
    }

    // Avoid division by zero: clamp window_sum to minimum value
    let eps = Tensor::<R>::full_scalar(&[1, full_len], real_dtype, 1e-8, device);
    let safe_window_sum = client.maximum(&window_sum, &eps)?;

    // Normalize by window sum (stays on device)
    let normalized_output = client.div(&output, &safe_window_sum)?;

    // Extract the final output region
    if pad_left == 0 && final_len == full_len {
        Ok(normalized_output)
    } else {
        // Extract [pad_left : pad_left + final_len] along last dimension
        let extracted =
            normalized_output.narrow(1, pad_left, final_len.min(full_len - pad_left))?;
        Ok(extracted.contiguous())
    }
}
