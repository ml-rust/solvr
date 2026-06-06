//! Generic convolution and correlation implementations.
//!
//! FFT-based 1D and 2D convolution/correlation using the overlap-save method.
use crate::DType;

use super::helpers::{complex_mul_impl, reverse_1d_impl, reverse_2d_impl};
use super::padding::{pad_1d_to_length_impl, pad_2d_to_shape_impl};
use super::slice::{slice_last_2d_impl, slice_last_dim_impl};
use crate::signal::{
    ConvMode, next_power_of_two, validate_kernel_1d, validate_kernel_2d, validate_signal_dtype,
};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of 1D convolution.
pub fn convolve_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    kernel: &Tensor<R>,
    mode: ConvMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let dtype = signal.dtype();
    validate_signal_dtype(dtype, "convolve")?;
    validate_kernel_1d(kernel.shape(), "convolve")?;

    if signal.dtype() != kernel.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: signal.dtype(),
            rhs: kernel.dtype(),
        });
    }

    let signal_contig = signal.contiguous()?;
    let kernel_contig = kernel.contiguous()?;

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
    let signal_padded = pad_1d_to_length_impl(client, &signal_contig, padded_len)?;
    let kernel_padded = pad_1d_to_length_impl(client, &kernel_contig, padded_len)?;

    // FFT both
    let signal_fft = client.rfft(&signal_padded, FftNormalization::None)?;
    let kernel_fft = client.rfft(&kernel_padded, FftNormalization::None)?;

    // Element-wise complex multiply
    let product = complex_mul_impl(client, &signal_fft, &kernel_fft)?;

    // Inverse FFT
    let result_full = client.irfft(&product, Some(padded_len), FftNormalization::Backward)?;

    // Slice to output size based on mode
    let output_len = mode.output_len(signal_len, kernel_len);
    let start = mode.slice_start(signal_len, kernel_len);

    slice_last_dim_impl(client, &result_full, start, output_len)
}

/// Generic implementation of 2D convolution.
pub fn convolve2d_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    kernel: &Tensor<R>,
    mode: ConvMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let dtype = signal.dtype();
    validate_signal_dtype(dtype, "convolve2d")?;
    validate_kernel_2d(kernel.shape(), "convolve2d")?;

    if signal.dtype() != kernel.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: signal.dtype(),
            rhs: kernel.dtype(),
        });
    }

    let signal_contig = signal.contiguous()?;
    let kernel_contig = kernel.contiguous()?;

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
    let signal_padded = pad_2d_to_shape_impl(client, &signal_contig, padded_h, padded_w)?;
    let kernel_padded = pad_2d_to_shape_impl(client, &kernel_contig, padded_h, padded_w)?;

    // 2D FFT both
    let signal_fft = client.rfft2(&signal_padded, FftNormalization::None)?;
    let kernel_fft = client.rfft2(&kernel_padded, FftNormalization::None)?;

    // Element-wise complex multiply
    let product = complex_mul_impl(client, &signal_fft, &kernel_fft)?;

    // Inverse 2D FFT
    let result_raw = client.irfft2(
        &product,
        Some((padded_h, padded_w)),
        FftNormalization::Backward,
    )?;

    // Apply missing normalization for first dimension
    let scale = 1.0 / (padded_h as f64);
    let result_full = client.mul_scalar(&result_raw, scale)?;

    // Slice to output size based on mode
    let (out_h, out_w) = mode.output_shape_2d((signal_h, signal_w), (kernel_h, kernel_w));
    let start_h = mode.slice_start(signal_h, kernel_h);
    let start_w = mode.slice_start(signal_w, kernel_w);

    slice_last_2d_impl(client, &result_full, start_h, out_h, start_w, out_w)
}

/// Generic implementation of 1D cross-correlation.
pub fn correlate_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    kernel: &Tensor<R>,
    mode: ConvMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Correlation is convolution with reversed kernel
    let kernel_reversed = reverse_1d_impl(client, kernel)?;
    convolve_impl(client, signal, &kernel_reversed, mode)
}

/// Generic implementation of 2D cross-correlation.
pub fn correlate2d_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    kernel: &Tensor<R>,
    mode: ConvMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Correlation is convolution with reversed kernel (flip both dims)
    let kernel_reversed = reverse_2d_impl(client, kernel)?;
    convolve2d_impl(client, signal, &kernel_reversed, mode)
}
