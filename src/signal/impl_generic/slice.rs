//! Generic slicing operations for signal processing.
//!
//! These functions extract slices from the last dimension(s) of tensors,
//! used for extracting convolution results based on output mode.
//!
//! All operations use numr's `narrow()` to keep data on device.
use crate::DType;

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Slice last dimension of tensor (generic over Runtime).
///
/// Extracts elements [start, start + len) from the last dimension.
/// Uses `narrow()` to keep all data on device (no CPU transfers).
/// Returns a contiguous tensor for compatibility with downstream operations.
pub fn slice_last_dim_impl<R, C>(
    _client: &C,
    tensor: &Tensor<R>,
    start: usize,
    len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    // Use narrow() on last dimension (-1) - data stays on device
    // Make contiguous for compatibility (still on-device, no CPU transfer)
    tensor.narrow(-1, start, len)?.contiguous()
}

/// Slice last two dimensions of tensor (generic over Runtime).
///
/// Extracts a rectangular region from the last two dimensions.
/// Uses `narrow()` twice to keep all data on device (no CPU transfers).
/// Returns a contiguous tensor for compatibility with downstream operations.
pub fn slice_last_2d_impl<R, C>(
    _client: &C,
    tensor: &Tensor<R>,
    start_h: usize,
    len_h: usize,
    start_w: usize,
    len_w: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    // Narrow dimension -2 (height), then dimension -1 (width)
    // Data stays on device throughout
    // Make contiguous for compatibility (still on-device, no CPU transfer)
    let sliced_h = tensor.narrow(-2, start_h, len_h)?;
    sliced_h.narrow(-1, start_w, len_w)?.contiguous()
}
