//! CPU STFT/ISTFT implementation.
//!
//! This module contains the CPU-specific STFT and ISTFT implementations that
//! use direct pointer access for maximum performance while delegating core
//! algorithm logic to the shared `stft_core` module.

// Allow many arguments for STFT implementation functions - these are internal
// functions where introducing a params struct would add unnecessary complexity.
#![allow(clippy::too_many_arguments)]

use crate::signal::stft_core::{
    extract_windowed_frame_f32, extract_windowed_frame_f64,
    normalize_and_copy_f32, normalize_and_copy_f64,
    overlap_add_f32, overlap_add_f64,
};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::{Complex128, Complex64};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

/// STFT implementation for F32.
pub(crate) fn stft_impl_f32(
    client: &CpuClient,
    signal: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    norm: FftNormalization,
) -> Result<()> {
    let signal_ptr = signal.storage().ptr() as *const f32;
    let window_ptr = window.storage().ptr() as *const f32;
    let output_ptr = output.storage().ptr() as *mut Complex64;

    let signal_len = signal.shape()[signal.ndim() - 1];
    let freq_bins = n_fft / 2 + 1;

    // SAFETY: signal is contiguous with signal_len elements per batch.
    // window is contiguous with n_fft elements.
    // Both are valid for the lifetime of this function.
    let window_slice =
        unsafe { std::slice::from_raw_parts(window_ptr, n_fft) };

    // Temporary buffer for windowed frame
    let mut frame = vec![0.0f32; n_fft];

    for b in 0..batch_size {
        // SAFETY: Each batch has signal_len elements.
        let signal_slice = unsafe {
            std::slice::from_raw_parts(signal_ptr.add(b * signal_len), signal_len)
        };
        let out_offset = b * n_frames * freq_bins;

        for f in 0..n_frames {
            let frame_start = f * hop;

            // Extract and window the frame using shared algorithm
            extract_windowed_frame_f32(signal_slice, window_slice, frame_start, &mut frame);

            // Create tensor for this frame and compute rfft
            let frame_tensor =
                Tensor::<CpuRuntime>::from_slice(&frame, &[n_fft], client.device());
            let spectrum = client.rfft(&frame_tensor, norm)?;

            // Copy spectrum to output
            // SAFETY: Output has batch_size * n_frames * freq_bins elements.
            // We're writing to position [b, f, :] which is at offset out_offset + f * freq_bins.
            let spec_ptr = spectrum.storage().ptr() as *const Complex64;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    spec_ptr,
                    output_ptr.add(out_offset + f * freq_bins),
                    freq_bins,
                );
            }
        }
    }

    Ok(())
}

/// STFT implementation for F64.
pub(crate) fn stft_impl_f64(
    client: &CpuClient,
    signal: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    norm: FftNormalization,
) -> Result<()> {
    let signal_ptr = signal.storage().ptr() as *const f64;
    let window_ptr = window.storage().ptr() as *const f64;
    let output_ptr = output.storage().ptr() as *mut Complex128;

    let signal_len = signal.shape()[signal.ndim() - 1];
    let freq_bins = n_fft / 2 + 1;

    // SAFETY: Same as stft_impl_f32.
    let window_slice =
        unsafe { std::slice::from_raw_parts(window_ptr, n_fft) };

    let mut frame = vec![0.0f64; n_fft];

    for b in 0..batch_size {
        let signal_slice = unsafe {
            std::slice::from_raw_parts(signal_ptr.add(b * signal_len), signal_len)
        };
        let out_offset = b * n_frames * freq_bins;

        for f in 0..n_frames {
            let frame_start = f * hop;
            extract_windowed_frame_f64(signal_slice, window_slice, frame_start, &mut frame);

            let frame_tensor =
                Tensor::<CpuRuntime>::from_slice(&frame, &[n_fft], client.device());
            let spectrum = client.rfft(&frame_tensor, norm)?;

            let spec_ptr = spectrum.storage().ptr() as *const Complex128;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    spec_ptr,
                    output_ptr.add(out_offset + f * freq_bins),
                    freq_bins,
                );
            }
        }
    }

    Ok(())
}

/// ISTFT implementation for F32.
pub(crate) fn istft_impl_f32(
    client: &CpuClient,
    stft_matrix: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    center: bool,
    final_len: usize,
    norm: FftNormalization,
) -> Result<()> {
    let stft_ptr = stft_matrix.storage().ptr() as *const Complex64;
    let window_ptr = window.storage().ptr() as *const f32;
    let output_ptr = output.storage().ptr() as *mut f32;

    let freq_bins = n_fft / 2 + 1;
    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };

    // SAFETY: Window tensor has n_fft elements.
    let window_slice =
        unsafe { std::slice::from_raw_parts(window_ptr, n_fft) };

    // Temporary buffers
    let mut reconstruction = vec![0.0f32; full_len];
    let mut window_sum = vec![0.0f32; full_len];

    for b in 0..batch_size {
        let stft_offset = b * n_frames * freq_bins;
        let out_offset = b * final_len;

        // Reset buffers
        reconstruction.fill(0.0);
        window_sum.fill(0.0);

        // Collect frames for overlap-add
        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);

        for f in 0..n_frames {
            // Get spectrum for this frame
            // SAFETY: STFT matrix has batch_size * n_frames * freq_bins elements.
            let frame_spectrum: Vec<Complex64> = (0..freq_bins)
                .map(|i| unsafe { *stft_ptr.add(stft_offset + f * freq_bins + i) })
                .collect();

            let spectrum_tensor =
                Tensor::<CpuRuntime>::from_slice(&frame_spectrum, &[freq_bins], client.device());

            // Inverse rfft
            let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
            let frame_data: Vec<f32> = frame.to_vec();
            frames.push(frame_data);
        }

        // Overlap-add using shared algorithm
        overlap_add_f32(
            frames.iter().enumerate().map(|(i, f)| (i, f.as_slice())),
            window_slice,
            &mut reconstruction,
            &mut window_sum,
            n_fft,
            hop,
        );

        // Normalize and copy to output using shared algorithm
        // SAFETY: Output has batch_size * final_len elements.
        let output_slice = unsafe {
            std::slice::from_raw_parts_mut(output_ptr.add(out_offset), final_len)
        };
        normalize_and_copy_f32(
            &reconstruction,
            &window_sum,
            output_slice,
            pad_left,
            1e-8,
        );
    }

    Ok(())
}

/// ISTFT implementation for F64.
pub(crate) fn istft_impl_f64(
    client: &CpuClient,
    stft_matrix: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    center: bool,
    final_len: usize,
    norm: FftNormalization,
) -> Result<()> {
    let stft_ptr = stft_matrix.storage().ptr() as *const Complex128;
    let window_ptr = window.storage().ptr() as *const f64;
    let output_ptr = output.storage().ptr() as *mut f64;

    let freq_bins = n_fft / 2 + 1;
    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };

    let window_slice =
        unsafe { std::slice::from_raw_parts(window_ptr, n_fft) };

    let mut reconstruction = vec![0.0f64; full_len];
    let mut window_sum = vec![0.0f64; full_len];

    for b in 0..batch_size {
        let stft_offset = b * n_frames * freq_bins;
        let out_offset = b * final_len;

        reconstruction.fill(0.0);
        window_sum.fill(0.0);

        let mut frames: Vec<Vec<f64>> = Vec::with_capacity(n_frames);

        for f in 0..n_frames {
            let frame_spectrum: Vec<Complex128> = (0..freq_bins)
                .map(|i| unsafe { *stft_ptr.add(stft_offset + f * freq_bins + i) })
                .collect();

            let spectrum_tensor =
                Tensor::<CpuRuntime>::from_slice(&frame_spectrum, &[freq_bins], client.device());
            let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
            let frame_data: Vec<f64> = frame.to_vec();
            frames.push(frame_data);
        }

        overlap_add_f64(
            frames.iter().enumerate().map(|(i, f)| (i, f.as_slice())),
            window_slice,
            &mut reconstruction,
            &mut window_sum,
            n_fft,
            hop,
        );

        let output_slice = unsafe {
            std::slice::from_raw_parts_mut(output_ptr.add(out_offset), final_len)
        };
        normalize_and_copy_f64(
            &reconstruction,
            &window_sum,
            output_slice,
            pad_left,
            1e-8,
        );
    }

    Ok(())
}
