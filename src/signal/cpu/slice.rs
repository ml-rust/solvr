//! CPU slicing operations for signal processing.
//!
//! This module provides functions to extract slices from the last dimension(s)
//! of tensors, used for extracting convolution results based on output mode.

use numr::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Slice last dimension of tensor.
///
/// Extracts elements [start, start + len) from the last dimension.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [..., src_len]
/// * `start` - Starting index in the last dimension
/// * `len` - Number of elements to extract
pub(crate) fn slice_last_dim(
    tensor: &Tensor<CpuRuntime>,
    start: usize,
    len: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = len;

    let output = Tensor::<CpuRuntime>::empty(&out_shape, dtype, tensor.storage().device());

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);
    let src_stride = tensor.shape()[ndim - 1];

    let src_ptr = tensor.storage().ptr();
    let dst_ptr = output.storage().ptr();
    let elem_size = dtype.size_in_bytes();

    // SAFETY: We're extracting a slice from each batch.
    // - Source: batch_size * src_stride elements
    // - Destination: batch_size * len elements
    // For each batch, we copy `len` elements starting at offset `start`.
    // Caller must ensure start + len <= src_stride.
    unsafe {
        for b in 0..batch_size {
            let src = (src_ptr as *const u8).add((b * src_stride + start) * elem_size);
            let dst = (dst_ptr as *mut u8).add(b * len * elem_size);
            std::ptr::copy_nonoverlapping(src, dst, len * elem_size);
        }
    }

    Ok(output)
}

/// Slice last two dimensions of tensor.
///
/// Extracts a rectangular region from the last two dimensions.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape [..., src_h, src_w]
/// * `start_h` - Starting row index
/// * `len_h` - Number of rows to extract
/// * `start_w` - Starting column index
/// * `len_w` - Number of columns to extract
pub(crate) fn slice_last_2d(
    tensor: &Tensor<CpuRuntime>,
    start_h: usize,
    len_h: usize,
    start_w: usize,
    len_w: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 2] = len_h;
    out_shape[ndim - 1] = len_w;

    let output = Tensor::<CpuRuntime>::empty(&out_shape, dtype, tensor.storage().device());

    let batch_size: usize = tensor.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);
    let src_h = tensor.shape()[ndim - 2];
    let src_w = tensor.shape()[ndim - 1];

    let src_ptr = tensor.storage().ptr();
    let dst_ptr = output.storage().ptr();
    let elem_size = dtype.size_in_bytes();

    // SAFETY: We're extracting a rectangular slice from each batch.
    // - Source: batch_size * src_h * src_w elements
    // - Destination: batch_size * len_h * len_w elements
    // For each row in the output, we copy `len_w` elements.
    // Caller must ensure start_h + len_h <= src_h and start_w + len_w <= src_w.
    unsafe {
        for b in 0..batch_size {
            for row in 0..len_h {
                let src_row = start_h + row;
                let src = (src_ptr as *const u8)
                    .add((b * src_h * src_w + src_row * src_w + start_w) * elem_size);
                let dst =
                    (dst_ptr as *mut u8).add((b * len_h * len_w + row * len_w) * elem_size);
                std::ptr::copy_nonoverlapping(src, dst, len_w * elem_size);
            }
        }
    }

    Ok(output)
}
