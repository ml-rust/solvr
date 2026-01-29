//! CPU implementation of signal processing algorithms.
//!
//! This module implements the [`SignalProcessingAlgorithms`] trait for CPU
//! using numr's FFT primitives.
//!
//! # Module Organization
//!
//! - `helpers` - Complex arithmetic, tensor reversal, magnitude computation
//! - `padding` - Zero-padding and reflect-padding operations
//! - `slice` - Tensor slicing operations
//! - `stft` - STFT/ISTFT implementations

mod helpers;
mod padding;
mod slice;
mod stft;

use super::{
    next_power_of_two, stft_num_frames, validate_kernel_1d, validate_kernel_2d,
    validate_signal_dtype, validate_stft_params, ConvMode, SignalProcessingAlgorithms,
};
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::ScalarOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

impl SignalProcessingAlgorithms<CpuRuntime> for CpuClient {
    fn convolve(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = signal.dtype();
        validate_signal_dtype(dtype, "convolve")?;
        validate_kernel_1d(kernel.shape(), "convolve")?;

        if signal.dtype() != kernel.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: signal.dtype(),
                rhs: kernel.dtype(),
            });
        }

        let signal_contig = signal.contiguous();
        let kernel_contig = kernel.contiguous();

        let ndim = signal_contig.ndim();
        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "signal",
                reason: "convolve requires at least 1D signal".to_string(),
            });
        }

        let signal_len = signal_contig.shape()[ndim - 1];
        let kernel_len = kernel_contig.shape()[0];

        if signal_len == 0 || kernel_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "signal/kernel",
                reason: "convolve requires non-empty signal and kernel".to_string(),
            });
        }

        // Calculate padded length for FFT (next power of 2)
        let full_len = signal_len + kernel_len - 1;
        let padded_len = next_power_of_two(full_len);

        // Pad signal and kernel to padded_len
        let signal_padded = padding::pad_1d_to_length(&signal_contig, padded_len, self.device())?;
        let kernel_padded = padding::pad_1d_to_length(&kernel_contig, padded_len, self.device())?;

        // FFT both
        let signal_fft = self.rfft(&signal_padded, FftNormalization::None)?;
        let kernel_fft = self.rfft(&kernel_padded, FftNormalization::None)?;

        // Element-wise complex multiply
        let product = helpers::complex_mul(&signal_fft, &kernel_fft)?;

        // Inverse FFT
        let result_full = self.irfft(&product, Some(padded_len), FftNormalization::Backward)?;

        // Slice to output size based on mode
        let output_len = mode.output_len(signal_len, kernel_len);
        let start = mode.slice_start(signal_len, kernel_len);

        slice::slice_last_dim(&result_full, start, output_len)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = signal.dtype();
        validate_signal_dtype(dtype, "convolve2d")?;
        validate_kernel_2d(kernel.shape(), "convolve2d")?;

        if signal.dtype() != kernel.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: signal.dtype(),
                rhs: kernel.dtype(),
            });
        }

        let signal_contig = signal.contiguous();
        let kernel_contig = kernel.contiguous();

        let ndim = signal_contig.ndim();
        if ndim < 2 {
            return Err(Error::InvalidArgument {
                arg: "signal",
                reason: "convolve2d requires at least 2D signal".to_string(),
            });
        }

        let signal_h = signal_contig.shape()[ndim - 2];
        let signal_w = signal_contig.shape()[ndim - 1];
        let kernel_h = kernel_contig.shape()[0];
        let kernel_w = kernel_contig.shape()[1];

        // Calculate padded dimensions
        let full_h = signal_h + kernel_h - 1;
        let full_w = signal_w + kernel_w - 1;
        let padded_h = next_power_of_two(full_h);
        let padded_w = next_power_of_two(full_w);

        // Pad signal and kernel
        let signal_padded =
            padding::pad_2d_to_shape(&signal_contig, padded_h, padded_w, self.device())?;
        let kernel_padded =
            padding::pad_2d_to_shape(&kernel_contig, padded_h, padded_w, self.device())?;

        // 2D FFT both
        let signal_fft = self.rfft2(&signal_padded, FftNormalization::None)?;
        let kernel_fft = self.rfft2(&kernel_padded, FftNormalization::None)?;

        // Element-wise complex multiply
        let product = helpers::complex_mul(&signal_fft, &kernel_fft)?;

        // Inverse 2D FFT
        let result_raw =
            self.irfft2(&product, Some((padded_h, padded_w)), FftNormalization::Backward)?;

        // Apply missing normalization for first dimension
        let scale = 1.0 / (padded_h as f64);
        let result_full = self.mul_scalar(&result_raw, scale)?;

        // Slice to output size based on mode
        let (out_h, out_w) = mode.output_shape_2d((signal_h, signal_w), (kernel_h, kernel_w));
        let start_h = mode.slice_start(signal_h, kernel_h);
        let start_w = mode.slice_start(signal_w, kernel_w);

        slice::slice_last_2d(&result_full, start_h, out_h, start_w, out_w)
    }

    fn correlate(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        // Correlation is convolution with reversed kernel
        let kernel_reversed = helpers::reverse_1d(kernel)?;
        self.convolve(signal, &kernel_reversed, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        // Correlation is convolution with reversed kernel (flip both dims)
        let kernel_reversed = helpers::reverse_2d(kernel)?;
        self.convolve2d(signal, &kernel_reversed, mode)
    }

    fn stft(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
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
            default_window = self.hann_window(n_fft, dtype, self.device())?;
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
            padding::pad_1d_reflect(&signal_contig, pad_left, pad_right, self.device())?
        } else {
            signal_contig.clone()
        };

        // Extract frames and apply window
        let batch_size: usize = if ndim > 1 {
            signal_contig.shape()[..ndim - 1].iter().product()
        } else {
            1
        };

        let freq_bins = n_fft / 2 + 1;
        let complex_dtype = match dtype {
            DType::F32 => DType::Complex64,
            DType::F64 => DType::Complex128,
            _ => unreachable!(),
        };

        // Output shape: [..., n_frames, freq_bins]
        let mut out_shape: Vec<usize> = signal_contig.shape()[..ndim - 1].to_vec();
        out_shape.push(n_frames);
        out_shape.push(freq_bins);

        let output = Tensor::<CpuRuntime>::empty(&out_shape, complex_dtype, self.device());

        // Process each batch
        let norm = if normalized {
            FftNormalization::Ortho
        } else {
            FftNormalization::Backward
        };

        match dtype {
            DType::F32 => {
                stft::stft_impl_f32(
                    self,
                    &padded_signal,
                    win,
                    &output,
                    n_fft,
                    hop,
                    n_frames,
                    batch_size,
                    norm,
                )?;
            }
            DType::F64 => {
                stft::stft_impl_f64(
                    self,
                    &padded_signal,
                    win,
                    &output,
                    n_fft,
                    hop,
                    n_frames,
                    batch_size,
                    norm,
                )?;
            }
            _ => unreachable!(),
        }

        Ok(output)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CpuRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
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
            default_window = self.hann_window(n_fft, real_dtype, self.device())?;
            &default_window
        };

        let batch_size: usize = if ndim > 2 {
            stft_contig.shape()[..ndim - 2].iter().product()
        } else {
            1
        };

        // Calculate output length
        let expected_len = n_fft + (n_frames - 1) * hop;
        let output_len = if center {
            expected_len - n_fft // Remove padding
        } else {
            expected_len
        };
        let final_len = length.unwrap_or(output_len);

        // Output shape: [..., final_len]
        let mut out_shape: Vec<usize> = stft_contig.shape()[..ndim - 2].to_vec();
        out_shape.push(final_len);

        let output = Tensor::<CpuRuntime>::zeros(&out_shape, real_dtype, self.device());

        let norm = if normalized {
            FftNormalization::Ortho
        } else {
            FftNormalization::Backward
        };

        match real_dtype {
            DType::F32 => {
                stft::istft_impl_f32(
                    self,
                    &stft_contig,
                    win,
                    &output,
                    n_fft,
                    hop,
                    n_frames,
                    batch_size,
                    center,
                    final_len,
                    norm,
                )?;
            }
            DType::F64 => {
                stft::istft_impl_f64(
                    self,
                    &stft_contig,
                    win,
                    &output,
                    n_fft,
                    hop,
                    n_frames,
                    batch_size,
                    center,
                    final_len,
                    norm,
                )?;
            }
            _ => unreachable!(),
        }

        Ok(output)
    }

    fn spectrogram(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        power: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        // Compute STFT
        let stft_result = self.stft(signal, n_fft, hop_length, window, true, false)?;

        // Compute magnitude^power
        let dtype = signal.dtype();
        helpers::complex_magnitude_pow(&stft_result, power, dtype)
    }
}
