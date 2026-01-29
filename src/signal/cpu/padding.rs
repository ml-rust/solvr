//! CPU padding operations for signal processing.
//!
//! This module provides padding functions used for FFT-based convolution and STFT.
//! All functions handle batched tensors where the last dimension(s) contain the signal.

use crate::signal::stft_core::reflect_pad_1d;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Pad 1D tensor to specified length (zero-padding at end).
///
/// Handles batched tensors: pads the last dimension to `target_len`.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [..., current_len]
/// * `target_len` - Target length for the last dimension
/// * `device` - CPU device for output tensor
pub(crate) fn pad_1d_to_length(
    tensor: &Tensor<CpuRuntime>,
    target_len: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    if current_len >= target_len {
        return Ok(tensor.clone());
    }

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    let output = Tensor::<CpuRuntime>::zeros(&out_shape, dtype, device);

    // Calculate batch size (product of all dimensions except last)
    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    let src_ptr = tensor.storage().ptr();
    let dst_ptr = output.storage().ptr();
    let elem_size = dtype.size_in_bytes();

    // SAFETY: We're copying data from a contiguous source tensor to a freshly
    // allocated output tensor. The loop bounds ensure we stay within both buffers:
    // - Source: batch_size * current_len elements (exactly the source tensor size)
    // - Destination: batch_size * target_len elements (exactly the output tensor size)
    // Each batch copies current_len elements, which is <= target_len.
    unsafe {
        for b in 0..batch_size {
            let src = (src_ptr as *const u8).add(b * current_len * elem_size);
            let dst = (dst_ptr as *mut u8).add(b * target_len * elem_size);
            std::ptr::copy_nonoverlapping(src, dst, current_len * elem_size);
        }
    }

    Ok(output)
}

/// Pad 2D tensor to specified shape (zero-padding).
///
/// Handles batched tensors: pads the last two dimensions.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [..., current_h, current_w]
/// * `target_h` - Target height
/// * `target_w` - Target width
/// * `device` - CPU device for output tensor
pub(crate) fn pad_2d_to_shape(
    tensor: &Tensor<CpuRuntime>,
    target_h: usize,
    target_w: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_h = tensor.shape()[ndim - 2];
    let current_w = tensor.shape()[ndim - 1];

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 2] = target_h;
    out_shape[ndim - 1] = target_w;

    let output = Tensor::<CpuRuntime>::zeros(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);

    let src_ptr = tensor.storage().ptr();
    let dst_ptr = output.storage().ptr();
    let elem_size = dtype.size_in_bytes();

    // SAFETY: We copy row-by-row from source to destination.
    // - Source has batch_size * current_h * current_w elements
    // - Destination has batch_size * target_h * target_w elements
    // For each row, we copy current_w elements (which is <= target_w).
    // Row indices range from 0 to current_h-1 (which is < target_h).
    unsafe {
        for b in 0..batch_size {
            for row in 0..current_h {
                let src = (src_ptr as *const u8)
                    .add((b * current_h * current_w + row * current_w) * elem_size);
                let dst = (dst_ptr as *mut u8)
                    .add((b * target_h * target_w + row * target_w) * elem_size);
                std::ptr::copy_nonoverlapping(src, dst, current_w * elem_size);
            }
        }
    }

    Ok(output)
}

/// Reflect padding for 1D signal (used in STFT centering).
///
/// Pads the signal by reflecting values at the boundaries.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [..., current_len]
/// * `pad_left` - Number of elements to pad on the left
/// * `pad_right` - Number of elements to pad on the right
/// * `device` - CPU device for output tensor
pub(crate) fn pad_1d_reflect(
    tensor: &Tensor<CpuRuntime>,
    pad_left: usize,
    pad_right: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    let target_len = current_len + pad_left + pad_right;
    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    let output = Tensor::<CpuRuntime>::empty(&out_shape, dtype, device);

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    match dtype {
        DType::F32 => reflect_pad_impl::<f32>(
            tensor,
            &output,
            pad_left,
            pad_right,
            current_len,
            target_len,
            batch_size,
        ),
        DType::F64 => reflect_pad_impl::<f64>(
            tensor,
            &output,
            pad_left,
            pad_right,
            current_len,
            target_len,
            batch_size,
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "pad_1d_reflect",
            })
        }
    }

    Ok(output)
}

/// Generic reflect padding implementation.
fn reflect_pad_impl<T: Clone + Copy>(
    tensor: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    pad_left: usize,
    pad_right: usize,
    current_len: usize,
    target_len: usize,
    batch_size: usize,
) {
    // SAFETY: We're accessing raw pointers to tensor storage.
    // - Source tensor has batch_size * current_len elements
    // - Output tensor has batch_size * target_len elements
    // The reflect_pad_1d function handles bounds checking for reflection indices.
    let src_ptr = tensor.storage().ptr() as *const T;
    let dst_ptr = output.storage().ptr() as *mut T;

    unsafe {
        for b in 0..batch_size {
            let src = std::slice::from_raw_parts(src_ptr.add(b * current_len), current_len);
            let dst = std::slice::from_raw_parts_mut(dst_ptr.add(b * target_len), target_len);
            reflect_pad_1d(src, dst, pad_left, pad_right);
        }
    }
}
