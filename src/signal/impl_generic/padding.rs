//! Generic padding operations for signal processing.
//!
//! These functions handle zero-padding and reflect-padding for FFT-based
//! convolution and STFT operations, using numr tensor operations.
//!
//! # Indexing Convention
//!
//! This module uses negative indexing (-1) consistently to refer to the last
//! dimension. This matches NumPy/PyTorch conventions and works correctly
//! regardless of the tensor's actual dimensionality.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pad 1D tensor to specified length - zero-padding at end using numr's pad().
pub fn pad_1d_to_length_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    target_len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    if current_len >= target_len {
        return Ok(tensor.contiguous());
    }

    let pad_right = target_len - current_len;
    // pad() uses pairs: [last_before, last_after]
    client.pad(tensor, &[0, pad_right], 0.0)
}

/// Pad 2D tensor to specified shape - zero-padding using numr's pad().
pub fn pad_2d_to_shape_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let ndim = tensor.ndim();
    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "pad_2d requires at least 2D tensor".to_string(),
        });
    }

    let current_h = tensor.shape()[ndim - 2];
    let current_w = tensor.shape()[ndim - 1];

    let pad_h = target_h.saturating_sub(current_h);
    let pad_w = target_w.saturating_sub(current_w);

    // pad() uses pairs from last dim: [last_before, last_after, second_last_before, second_last_after]
    client.pad(tensor, &[0, pad_w, 0, pad_h], 0.0)
}

/// Reflect padding for 1D signal - used in STFT centering.
///
/// Uses tensor slicing and concatenation to build the padded result on device.
pub fn pad_1d_reflect_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    pad_left: usize,
    pad_right: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "pad_1d_reflect",
        });
    }

    let ndim = tensor.ndim();
    let current_len = tensor.shape()[ndim - 1];

    if current_len == 0 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "Cannot reflect-pad empty tensor".to_string(),
        });
    }

    // Handle edge case: tensor too small for requested padding
    if pad_left >= current_len || pad_right >= current_len {
        return Err(Error::InvalidArgument {
            arg: "padding",
            reason: format!(
                "Reflect padding ({}, {}) too large for tensor length {}",
                pad_left, pad_right, current_len
            ),
        });
    }

    if pad_left == 0 && pad_right == 0 {
        return Ok(tensor.contiguous());
    }

    // Build parts to concatenate
    let mut parts: Vec<Tensor<R>> = Vec::new();

    // Left reflection: tensor[1:pad_left+1] reversed
    if pad_left > 0 {
        let left_slice = tensor.narrow(-1, 1, pad_left)?.contiguous();
        let left_reflected = left_slice.flip(-1)?;
        parts.push(left_reflected);
    }

    // Original tensor
    parts.push(tensor.contiguous());

    // Right reflection: tensor[len-pad_right-1:len-1] reversed
    if pad_right > 0 {
        let start = current_len - pad_right - 1;
        let right_slice = tensor.narrow(-1, start, pad_right)?.contiguous();
        let right_reflected = right_slice.flip(-1)?;
        parts.push(right_reflected);
    }

    // Concatenate along last dimension
    let refs: Vec<&Tensor<R>> = parts.iter().collect();
    client.cat(&refs, -1)
}
