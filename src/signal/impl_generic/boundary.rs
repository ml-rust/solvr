//! Boundary mode padding for N-D filters.
//!
//! Implements padding strategies for different boundary modes used by
//! scipy.ndimage-equivalent filters. All operations stay on device - no GPU->CPU transfers.
use crate::DType;

use crate::signal::traits::nd_filters::BoundaryMode;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pad a tensor along a single axis according to the boundary mode.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `input` - Input tensor
/// * `axis` - Axis to pad (supports negative indexing)
/// * `pad_before` - Number of elements to pad before
/// * `pad_after` - Number of elements to pad after
/// * `mode` - Boundary handling mode
///
/// # Returns
///
/// Padded tensor with same ndim as input, but expanded along axis
pub fn pad_axis_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    axis: isize,
    pad_before: usize,
    pad_after: usize,
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    if pad_before == 0 && pad_after == 0 {
        return input.contiguous();
    }

    let ndim = input.ndim() as isize;
    // Normalize axis to positive index
    let axis_normalized = if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    };

    if axis_normalized >= input.ndim() {
        return Err(Error::InvalidArgument {
            arg: "axis",
            reason: format!(
                "axis {} out of range for tensor with ndim {}",
                axis,
                input.ndim()
            ),
        });
    }

    let axis_len = input.shape()[axis_normalized];

    match mode {
        BoundaryMode::Constant(value) => {
            // Use numr's pad op with all-zero padding except for the target axis
            // padding format: [last_before, last_after, second_last_before, second_last_after, ...]
            let mut padding = vec![0usize; input.ndim() * 2];
            // Convert axis to dimension index from the right (numr's pad uses right-to-left order)
            let dim_idx = input.ndim() - axis_normalized - 1;
            padding[dim_idx * 2] = pad_before;
            padding[dim_idx * 2 + 1] = pad_after;

            client.pad(input, &padding, value)
        }
        BoundaryMode::Reflect => {
            // Reflect: d c b a | a b c d | d c b a (half-sample symmetric)
            // Take slices from the input, flip them, and concatenate
            let mut parts: Vec<Tensor<R>> = Vec::new();

            if pad_before > 0 {
                // Reflect from beginning: take elements [1..min(pad_before+1, axis_len)]
                let take = pad_before.min(axis_len.saturating_sub(1));
                if take > 0 {
                    let slice = input.narrow(axis, 1, take)?;
                    let flipped = slice.flip(axis)?;
                    parts.push(flipped);
                }
            }

            parts.push(input.contiguous()?);

            if pad_after > 0 {
                // Reflect from end: take last `pad_after` elements (excluding boundary), flip
                let take = pad_after.min(axis_len.saturating_sub(1));
                if take > 0 {
                    let start = axis_len.saturating_sub(take + 1);
                    let slice = input.narrow(axis, start, take)?;
                    let flipped = slice.flip(axis)?;
                    parts.push(flipped);
                }
            }

            // Concatenate along axis
            if parts.is_empty() {
                Ok(input.contiguous()?)
            } else {
                let refs: Vec<&Tensor<R>> = parts.iter().collect();
                client.cat(&refs, axis)
            }
        }
        BoundaryMode::Nearest => {
            // Nearest: a a a a | a b c d | d d d d (edge value repetition)
            let mut parts: Vec<Tensor<R>> = Vec::new();

            if pad_before > 0 {
                // Take first element along axis, repeat pad_before times
                let first = input.narrow(axis, 0, 1)?;
                let mut repeat_shape = vec![1usize; input.ndim()];
                repeat_shape[axis_normalized] = pad_before;
                let repeated = client.repeat(&first, &repeat_shape)?;
                parts.push(repeated);
            }

            parts.push(input.contiguous()?);

            if pad_after > 0 {
                // Take last element along axis, repeat pad_after times
                let last = input.narrow(axis, axis_len - 1, 1)?;
                let mut repeat_shape = vec![1usize; input.ndim()];
                repeat_shape[axis_normalized] = pad_after;
                let repeated = client.repeat(&last, &repeat_shape)?;
                parts.push(repeated);
            }

            if parts.is_empty() {
                Ok(input.contiguous()?)
            } else {
                let refs: Vec<&Tensor<R>> = parts.iter().collect();
                client.cat(&refs, axis)
            }
        }
        BoundaryMode::Mirror => {
            // Mirror: d c b | a b c d | c b a (whole-sample symmetric)
            // Like Reflect but includes the boundary element
            let mut parts: Vec<Tensor<R>> = Vec::new();

            if pad_before > 0 {
                let take = pad_before.min(axis_len);
                if take > 0 {
                    let slice = input.narrow(axis, 0, take)?;
                    let flipped = slice.flip(axis)?;
                    parts.push(flipped);
                }
            }

            parts.push(input.contiguous()?);

            if pad_after > 0 {
                let take = pad_after.min(axis_len);
                if take > 0 {
                    let start = axis_len.saturating_sub(take);
                    let slice = input.narrow(axis, start, take)?;
                    let flipped = slice.flip(axis)?;
                    parts.push(flipped);
                }
            }

            if parts.is_empty() {
                Ok(input.contiguous()?)
            } else {
                let refs: Vec<&Tensor<R>> = parts.iter().collect();
                client.cat(&refs, axis)
            }
        }
        BoundaryMode::Wrap => {
            // Wrap: a b c d | a b c d | a b c d (periodic/circular)
            let mut parts: Vec<Tensor<R>> = Vec::new();

            if pad_before > 0 {
                // Take from the end: elements [max(0, len - pad_before)..len]
                let take = pad_before.min(axis_len);
                if take > 0 {
                    let start = axis_len - take;
                    let slice = input.narrow(axis, start, take)?;
                    parts.push(slice);
                }
            }

            parts.push(input.contiguous()?);

            if pad_after > 0 {
                // Take from the beginning: elements [0..min(pad_after, len)]
                let take = pad_after.min(axis_len);
                if take > 0 {
                    let slice = input.narrow(axis, 0, take)?;
                    parts.push(slice);
                }
            }

            if parts.is_empty() {
                Ok(input.contiguous()?)
            } else {
                let refs: Vec<&Tensor<R>> = parts.iter().collect();
                client.cat(&refs, axis)
            }
        }
    }
}
