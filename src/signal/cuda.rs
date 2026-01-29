//! CUDA implementation of signal processing algorithms
//!
//! This module implements the [`SignalProcessingAlgorithms`] trait for CUDA
//! using numr's FFT primitives.

use super::{
    next_power_of_two, stft_num_frames, validate_kernel_1d, validate_kernel_2d,
    validate_signal_dtype, validate_stft_params, ConvMode, SignalProcessingAlgorithms,
};
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::{Complex128, Complex64, DType};
use numr::error::{Error, Result};
use numr::ops::ScalarOps;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Layout, Storage, Tensor};

// ============================================================================
// SignalProcessingAlgorithms Implementation
// ============================================================================

impl SignalProcessingAlgorithms<CudaRuntime> for CudaClient {
    fn convolve(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let device = self.device();

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

        // Calculate padded length for FFT
        let full_len = signal_len + kernel_len - 1;
        let padded_len = next_power_of_two(full_len);

        // Pad signal and kernel
        let signal_padded = pad_1d_to_length_cuda(&signal_contig, padded_len, device, self)?;
        let kernel_padded = pad_1d_to_length_cuda(&kernel_contig, padded_len, device, self)?;

        // FFT both
        let signal_fft = self.rfft(&signal_padded, FftNormalization::None)?;
        let kernel_fft = self.rfft(&kernel_padded, FftNormalization::None)?;

        // Element-wise complex multiply
        let product = complex_mul_cuda(&signal_fft, &kernel_fft, device, self)?;

        // Inverse FFT
        let result_full = self.irfft(&product, Some(padded_len), FftNormalization::Backward)?;

        // Slice to output size
        let output_len = mode.output_len(signal_len, kernel_len);
        let start = mode.slice_start(signal_len, kernel_len);

        slice_last_dim_cuda(&result_full, start, output_len, device, self)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let device = self.device();

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
            pad_2d_to_shape_cuda(&signal_contig, padded_h, padded_w, device, self)?;
        let kernel_padded =
            pad_2d_to_shape_cuda(&kernel_contig, padded_h, padded_w, device, self)?;

        // 2D FFT both
        let signal_fft = self.rfft2(&signal_padded, FftNormalization::None)?;
        let kernel_fft = self.rfft2(&kernel_padded, FftNormalization::None)?;

        // Element-wise complex multiply
        let product = complex_mul_cuda(&signal_fft, &kernel_fft, device, self)?;

        // Inverse 2D FFT
        let result_raw =
            self.irfft2(&product, Some((padded_h, padded_w)), FftNormalization::Backward)?;

        // Apply missing normalization for first dimension
        let scale = 1.0 / (padded_h as f64);
        let result_full = self.mul_scalar(&result_raw, scale)?;

        // Slice to output size
        let (out_h, out_w) = mode.output_shape_2d((signal_h, signal_w), (kernel_h, kernel_w));
        let start_h = mode.slice_start(signal_h, kernel_h);
        let start_w = mode.slice_start(signal_w, kernel_w);

        slice_last_2d_cuda(&result_full, start_h, out_h, start_w, out_w, device, self)
    }

    fn correlate(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        let kernel_reversed = reverse_1d_cuda(kernel, self.device(), self)?;
        self.convolve(signal, &kernel_reversed, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        let kernel_reversed = reverse_2d_cuda(kernel, self.device(), self)?;
        self.convolve2d(signal, &kernel_reversed, mode)
    }

    fn stft(
        &self,
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = signal.dtype();
        validate_signal_dtype(dtype, "stft")?;

        let hop = hop_length.unwrap_or(n_fft / 4);
        validate_stft_params(n_fft, hop, "stft")?;

        let signal_contig = signal.contiguous();
        let device = self.device();
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
            default_window = self.hann_window(n_fft, dtype, device)?;
            &default_window
        };

        let n_frames = stft_num_frames(signal_len, n_fft, hop, center);

        if n_frames == 0 {
            return Err(Error::InvalidArgument {
                arg: "signal",
                reason: format!(
                    "signal too short for STFT: length={signal_len}, n_fft={n_fft}"
                ),
            });
        }

        // Pad signal if centering
        let padded_signal = if center {
            let pad_left = n_fft / 2;
            let pad_right = n_fft / 2;
            pad_1d_reflect_cuda(&signal_contig, pad_left, pad_right, device, self)?
        } else {
            signal_contig.clone()
        };

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

        let norm = if normalized {
            FftNormalization::Ortho
        } else {
            FftNormalization::Backward
        };

        // Process frame by frame
        stft_impl_cuda(
            self,
            &padded_signal,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            complex_dtype,
            norm,
            device,
        )
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CudaRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let device = self.device();
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
            default_window = self.hann_window(n_fft, real_dtype, device)?;
            &default_window
        };

        let batch_size: usize = if ndim > 2 {
            stft_contig.shape()[..ndim - 2].iter().product()
        } else {
            1
        };

        let expected_len = n_fft + (n_frames - 1) * hop;
        let output_len = if center {
            expected_len - n_fft
        } else {
            expected_len
        };
        let final_len = length.unwrap_or(output_len);

        let mut out_shape: Vec<usize> = stft_contig.shape()[..ndim - 2].to_vec();
        out_shape.push(final_len);

        let norm = if normalized {
            FftNormalization::Ortho
        } else {
            FftNormalization::Backward
        };

        istft_impl_cuda(
            self,
            &stft_contig,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            center,
            final_len,
            real_dtype,
            norm,
            device,
        )
    }

    fn spectrogram(
        &self,
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        power: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let stft_result = self.stft(signal, n_fft, hop_length, window, true, false)?;
        let dtype = signal.dtype();
        complex_magnitude_pow_cuda(&stft_result, power, dtype, self.device(), self)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn pad_1d_to_length_cuda(
    tensor: &Tensor<CudaRuntime>,
    target_len: usize,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    if current_len >= target_len {
        return Ok(tensor.clone());
    }

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    let output = Tensor::<CudaRuntime>::zeros(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);
    let elem_size = dtype.size_in_bytes();

    // Copy original data batch by batch
    for b in 0..batch_size {
        let src_offset = b * current_len * elem_size;
        let dst_offset = b * target_len * elem_size;
        CudaRuntime::copy_within_device(
            tensor.storage().ptr() + src_offset as u64,
            output.storage().ptr() + dst_offset as u64,
            current_len * elem_size,
            device,
        );
    }

    Ok(output)
}

fn pad_2d_to_shape_cuda(
    tensor: &Tensor<CudaRuntime>,
    target_h: usize,
    target_w: usize,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_h = tensor.shape()[ndim - 2];
    let current_w = tensor.shape()[ndim - 1];

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 2] = target_h;
    out_shape[ndim - 1] = target_w;

    let output = Tensor::<CudaRuntime>::zeros(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);
    let elem_size = dtype.size_in_bytes();

    // Copy row by row
    for b in 0..batch_size {
        for row in 0..current_h {
            let src_offset = (b * current_h * current_w + row * current_w) * elem_size;
            let dst_offset = (b * target_h * target_w + row * target_w) * elem_size;
            CudaRuntime::copy_within_device(
                tensor.storage().ptr() + src_offset as u64,
                output.storage().ptr() + dst_offset as u64,
                current_w * elem_size,
                device,
            );
        }
    }

    Ok(output)
}

fn pad_1d_reflect_cuda(
    tensor: &Tensor<CudaRuntime>,
    pad_left: usize,
    pad_right: usize,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    let target_len = current_len + pad_left + pad_right;
    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    // Transfer to CPU, pad, and transfer back
    let tensor_vec: Vec<u8> = tensor.to_vec::<u8>();
    let elem_size = dtype.size_in_bytes();

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    let mut output_vec = vec![0u8; batch_size * target_len * elem_size];

    for b in 0..batch_size {
        let src_start = b * current_len * elem_size;
        let dst_start = b * target_len * elem_size;

        // Left padding (reflected)
        for i in 0..pad_left {
            let reflect_idx = (pad_left - i).min(current_len - 1);
            let src_off = src_start + reflect_idx * elem_size;
            let dst_off = dst_start + i * elem_size;
            output_vec[dst_off..dst_off + elem_size]
                .copy_from_slice(&tensor_vec[src_off..src_off + elem_size]);
        }

        // Original data
        let src_off = src_start;
        let dst_off = dst_start + pad_left * elem_size;
        output_vec[dst_off..dst_off + current_len * elem_size]
            .copy_from_slice(&tensor_vec[src_off..src_off + current_len * elem_size]);

        // Right padding (reflected)
        for i in 0..(target_len - pad_left - current_len) {
            let reflect_idx = current_len.saturating_sub(2).saturating_sub(i).max(0);
            let src_off = src_start + reflect_idx * elem_size;
            let dst_off = dst_start + (pad_left + current_len + i) * elem_size;
            output_vec[dst_off..dst_off + elem_size]
                .copy_from_slice(&tensor_vec[src_off..src_off + elem_size]);
        }
    }

    let storage = Storage::from_bytes(&output_vec, dtype, device);
    let layout = Layout::contiguous(&out_shape);
    Ok(Tensor::<CudaRuntime>::from_parts(storage, layout))
}

fn slice_last_dim_cuda(
    tensor: &Tensor<CudaRuntime>,
    start: usize,
    len: usize,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = len;

    let output = Tensor::<CudaRuntime>::empty(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);
    let src_stride = tensor.shape()[ndim - 1];
    let elem_size = dtype.size_in_bytes();

    for b in 0..batch_size {
        let src_offset = (b * src_stride + start) * elem_size;
        let dst_offset = b * len * elem_size;
        CudaRuntime::copy_within_device(
            tensor.storage().ptr() + src_offset as u64,
            output.storage().ptr() + dst_offset as u64,
            len * elem_size,
            device,
        );
    }

    Ok(output)
}

fn slice_last_2d_cuda(
    tensor: &Tensor<CudaRuntime>,
    start_h: usize,
    len_h: usize,
    start_w: usize,
    len_w: usize,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 2] = len_h;
    out_shape[ndim - 1] = len_w;

    let output = Tensor::<CudaRuntime>::empty(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);
    let src_h = tensor.shape()[ndim - 2];
    let src_w = tensor.shape()[ndim - 1];
    let elem_size = dtype.size_in_bytes();

    for b in 0..batch_size {
        for row in 0..len_h {
            let src_row = start_h + row;
            let src_offset = (b * src_h * src_w + src_row * src_w + start_w) * elem_size;
            let dst_offset = (b * len_h * len_w + row * len_w) * elem_size;
            CudaRuntime::copy_within_device(
                tensor.storage().ptr() + src_offset as u64,
                output.storage().ptr() + dst_offset as u64,
                len_w * elem_size,
                device,
            );
        }
    }

    Ok(output)
}

fn reverse_1d_cuda(
    tensor: &Tensor<CudaRuntime>,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_1d requires 1D tensor".to_string(),
        });
    }

    let len = tensor_contig.shape()[0];

    // Transfer to CPU, reverse, transfer back
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let reversed: Vec<f32> = data.into_iter().rev().collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&reversed, &[len], device))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let reversed: Vec<f64> = data.into_iter().rev().collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&reversed, &[len], device))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "reverse_1d",
        }),
    }
}

fn reverse_2d_cuda(
    tensor: &Tensor<CudaRuntime>,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_2d requires 2D tensor".to_string(),
        });
    }

    let h = tensor_contig.shape()[0];
    let w = tensor_contig.shape()[1];

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut reversed = vec![0.0f32; h * w];
            for i in 0..h {
                for j in 0..w {
                    reversed[i * w + j] = data[(h - 1 - i) * w + (w - 1 - j)];
                }
            }
            Ok(Tensor::<CudaRuntime>::from_slice(&reversed, &[h, w], device))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut reversed = vec![0.0f64; h * w];
            for i in 0..h {
                for j in 0..w {
                    reversed[i * w + j] = data[(h - 1 - i) * w + (w - 1 - j)];
                }
            }
            Ok(Tensor::<CudaRuntime>::from_slice(&reversed, &[h, w], device))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "reverse_2d",
        }),
    }
}

fn complex_mul_cuda(
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    // Transfer to CPU, multiply, transfer back
    match dtype {
        DType::Complex64 => {
            let a_data: Vec<Complex64> = a.to_vec();
            let b_data: Vec<Complex64> = b.to_vec();
            let result: Vec<Complex64> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(av, bv)| {
                    Complex64::new(
                        av.re * bv.re - av.im * bv.im,
                        av.re * bv.im + av.im * bv.re,
                    )
                })
                .collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&result, a.shape(), device))
        }
        DType::Complex128 => {
            let a_data: Vec<Complex128> = a.to_vec();
            let b_data: Vec<Complex128> = b.to_vec();
            let result: Vec<Complex128> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(av, bv)| {
                    Complex128::new(
                        av.re * bv.re - av.im * bv.im,
                        av.re * bv.im + av.im * bv.re,
                    )
                })
                .collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&result, a.shape(), device))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "complex_mul",
        }),
    }
}

fn complex_magnitude_pow_cuda(
    tensor: &Tensor<CudaRuntime>,
    power: f64,
    output_dtype: DType,
    device: &CudaDevice,
    _client: &CudaClient,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = tensor.dtype();

    match (dtype, output_dtype) {
        (DType::Complex64, DType::F32) => {
            let data: Vec<Complex64> = tensor.to_vec();
            let result: Vec<f32> = data
                .iter()
                .map(|c| {
                    let mag_sq = c.re * c.re + c.im * c.im;
                    if power == 2.0 {
                        mag_sq
                    } else if power == 1.0 {
                        mag_sq.sqrt()
                    } else {
                        mag_sq.powf(power as f32 / 2.0)
                    }
                })
                .collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&result, tensor.shape(), device))
        }
        (DType::Complex128, DType::F64) => {
            let data: Vec<Complex128> = tensor.to_vec();
            let result: Vec<f64> = data
                .iter()
                .map(|c| {
                    let mag_sq = c.re * c.re + c.im * c.im;
                    if power == 2.0 {
                        mag_sq
                    } else if power == 1.0 {
                        mag_sq.sqrt()
                    } else {
                        mag_sq.powf(power / 2.0)
                    }
                })
                .collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&result, tensor.shape(), device))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "complex_magnitude_pow",
        }),
    }
}

// ============================================================================
// STFT/ISTFT Implementation
// ============================================================================

fn stft_impl_cuda(
    client: &CudaClient,
    signal: &Tensor<CudaRuntime>,
    window: &Tensor<CudaRuntime>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    complex_dtype: DType,
    norm: FftNormalization,
    device: &CudaDevice,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = signal.dtype();
    let signal_len = signal.shape()[signal.ndim() - 1];

    // Process frame by frame
    match dtype {
        DType::F32 => {
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
                    let frame_tensor =
                        Tensor::<CudaRuntime>::from_slice(&frame, &[n_fft], device);
                    let spectrum = client.rfft(&frame_tensor, norm)?;
                    let spec_data: Vec<Complex64> = spectrum.to_vec();

                    // Copy to output
                    let out_offset = b * n_frames * freq_bins + f * freq_bins;
                    output_data[out_offset..out_offset + freq_bins]
                        .copy_from_slice(&spec_data[..freq_bins]);
                }
            }

            Ok(Tensor::<CudaRuntime>::from_slice(&output_data, out_shape, device))
        }
        DType::F64 => {
            let signal_data: Vec<f64> = signal.to_vec();
            let window_data: Vec<f64> = window.to_vec();
            let mut output_data =
                vec![Complex128::new(0.0, 0.0); batch_size * n_frames * freq_bins];

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

                    let frame_tensor =
                        Tensor::<CudaRuntime>::from_slice(&frame, &[n_fft], device);
                    let spectrum = client.rfft(&frame_tensor, norm)?;
                    let spec_data: Vec<Complex128> = spectrum.to_vec();

                    let out_offset = b * n_frames * freq_bins + f * freq_bins;
                    output_data[out_offset..out_offset + freq_bins]
                        .copy_from_slice(&spec_data[..freq_bins]);
                }
            }

            Ok(Tensor::<CudaRuntime>::from_slice(&output_data, out_shape, device))
        }
        _ => Err(Error::UnsupportedDType { dtype, op: "stft" }),
    }
}

fn istft_impl_cuda(
    client: &CudaClient,
    stft_matrix: &Tensor<CudaRuntime>,
    window: &Tensor<CudaRuntime>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    center: bool,
    final_len: usize,
    real_dtype: DType,
    norm: FftNormalization,
    device: &CudaDevice,
) -> Result<Tensor<CudaRuntime>> {
    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };

    match real_dtype {
        DType::F32 => {
            let stft_data: Vec<Complex64> = stft_matrix.to_vec();
            let window_data: Vec<f32> = window.to_vec();
            let mut output_data = vec![0.0f32; batch_size * final_len];

            for b in 0..batch_size {
                let stft_offset = b * n_frames * freq_bins;
                let out_offset = b * final_len;

                let mut reconstruction = vec![0.0f32; full_len];
                let mut window_sum = vec![0.0f32; full_len];

                for f in 0..n_frames {
                    let frame_spectrum: Vec<Complex64> = stft_data
                        [stft_offset + f * freq_bins..stft_offset + (f + 1) * freq_bins]
                        .to_vec();

                    let spectrum_tensor =
                        Tensor::<CudaRuntime>::from_slice(&frame_spectrum, &[freq_bins], device);
                    let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
                    let frame_data: Vec<f32> = frame.to_vec();

                    let frame_start = f * hop;
                    for i in 0..n_fft {
                        let out_idx = frame_start + i;
                        if out_idx < full_len {
                            let win_val = window_data[i];
                            reconstruction[out_idx] += frame_data[i] * win_val;
                            window_sum[out_idx] += win_val * win_val;
                        }
                    }
                }

                for i in 0..final_len {
                    let src_idx = pad_left + i;
                    if src_idx < full_len {
                        let norm_factor = if window_sum[src_idx] > 1e-8 {
                            window_sum[src_idx]
                        } else {
                            1.0
                        };
                        output_data[out_offset + i] = reconstruction[src_idx] / norm_factor;
                    }
                }
            }

            Ok(Tensor::<CudaRuntime>::from_slice(&output_data, out_shape, device))
        }
        DType::F64 => {
            let stft_data: Vec<Complex128> = stft_matrix.to_vec();
            let window_data: Vec<f64> = window.to_vec();
            let mut output_data = vec![0.0f64; batch_size * final_len];

            for b in 0..batch_size {
                let stft_offset = b * n_frames * freq_bins;
                let out_offset = b * final_len;

                let mut reconstruction = vec![0.0f64; full_len];
                let mut window_sum = vec![0.0f64; full_len];

                for f in 0..n_frames {
                    let frame_spectrum: Vec<Complex128> = stft_data
                        [stft_offset + f * freq_bins..stft_offset + (f + 1) * freq_bins]
                        .to_vec();

                    let spectrum_tensor =
                        Tensor::<CudaRuntime>::from_slice(&frame_spectrum, &[freq_bins], device);
                    let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
                    let frame_data: Vec<f64> = frame.to_vec();

                    let frame_start = f * hop;
                    for i in 0..n_fft {
                        let out_idx = frame_start + i;
                        if out_idx < full_len {
                            let win_val = window_data[i];
                            reconstruction[out_idx] += frame_data[i] * win_val;
                            window_sum[out_idx] += win_val * win_val;
                        }
                    }
                }

                for i in 0..final_len {
                    let src_idx = pad_left + i;
                    if src_idx < full_len {
                        let norm_factor = if window_sum[src_idx] > 1e-8 {
                            window_sum[src_idx]
                        } else {
                            1.0
                        };
                        output_data[out_offset + i] = reconstruction[src_idx] / norm_factor;
                    }
                }
            }

            Ok(Tensor::<CudaRuntime>::from_slice(&output_data, out_shape, device))
        }
        _ => Err(Error::UnsupportedDType {
            dtype: real_dtype,
            op: "istft",
        }),
    }
}
