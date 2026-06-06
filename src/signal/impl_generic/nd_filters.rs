//! Generic N-dimensional filter implementations.
//!
//! Separable filters (Gaussian, uniform) are decomposed into sequential 1D
//! passes along each axis, using numr's conv1d in (N,C,L) format.
//! Min/max/percentile filters use a separable approach with rank operations.
use crate::DType;

use super::boundary::pad_axis_impl;
use super::kernels::{gaussian_kernel_1d, uniform_kernel_1d};
use crate::signal::traits::nd_filters::BoundaryMode;
use crate::signal::validate_signal_dtype;
use numr::error::{Error, Result};
use numr::ops::{ConvOps, PaddingMode, ReduceOps, ScalarOps, ShapeOps, SortingOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Expand size array to ndim length (repeat last element if shorter).
fn expand_sizes(sizes: &[usize], ndim: usize) -> Vec<usize> {
    if sizes.is_empty() {
        vec![1; ndim]
    } else {
        let mut result = sizes[..sizes.len().min(ndim)].to_vec();
        result.resize(ndim, result[result.len() - 1]);
        result
    }
}

/// Expand sigma array to ndim length.
fn expand_sigmas(sigmas: &[f64], ndim: usize) -> Vec<f64> {
    if sigmas.is_empty() {
        vec![1.0; ndim]
    } else {
        let mut result = sigmas[..sigmas.len().min(ndim)].to_vec();
        result.resize(ndim, result[result.len() - 1]);
        result
    }
}

/// Apply a 1D convolution along a specific axis of an N-D tensor.
///
/// Reshapes the tensor so that the target axis becomes the last dimension
/// of a 3D tensor (N,C,L), applies conv1d, then reshapes back.
fn convolve_along_axis<R, C>(
    client: &C,
    input: &Tensor<R>,
    kernel_1d: &Tensor<R>,
    axis: usize,
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ScalarOps<R> + ShapeOps<R> + RuntimeClient<R>,
{
    let shape = input.shape().to_vec();
    let ndim = shape.len();
    let kernel_len = kernel_1d.shape()[0];
    let pad_size = kernel_len / 2;

    // 1. Pad along the target axis with boundary mode
    let padded = pad_axis_impl(client, input, axis as isize, pad_size, pad_size, mode)?;
    let padded_shape = padded.shape().to_vec();

    // 2. Reshape to (batch, 1, axis_len) for conv1d
    // Move target axis to last position by transposing
    let axis_len = padded_shape[axis];
    let batch_size: usize = padded_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .product();

    // Permute to move axis to last
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.remove(axis);
    perm.push(axis);

    let permuted = padded.permute(&perm)?;
    let permuted_contig = permuted.contiguous()?;

    // Reshape to (batch, 1, axis_len)
    let reshaped = permuted_contig.reshape(&[batch_size, 1, axis_len])?;

    // Reshape kernel to (1, 1, kernel_len) for conv1d
    let kernel_3d = kernel_1d.reshape(&[1, 1, kernel_len])?;

    // 3. Apply conv1d with Valid padding (we already padded)
    let conv_result = client.conv1d(&reshaped, &kernel_3d, None, 1, PaddingMode::Valid, 1, 1)?;

    // 4. Reshape back
    // conv_result shape: (batch, 1, output_len) where output_len = original axis_len
    let output_axis_len = shape[axis]; // original size
    let mut permuted_shape: Vec<usize> = perm[..ndim - 1].iter().map(|&i| shape[i]).collect();
    permuted_shape.push(output_axis_len);

    let reshaped_back = conv_result.reshape(&permuted_shape)?;

    // Inverse permute to restore original axis order
    let mut inv_perm = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    let result = reshaped_back.permute(&inv_perm)?;

    result.contiguous()
}

/// Generic Gaussian filter implementation.
pub fn gaussian_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    sigma: &[f64],
    order: &[usize],
    mode: BoundaryMode,
    truncate: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ScalarOps<R> + ShapeOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let dtype = input.dtype();
    validate_signal_dtype(dtype, "gaussian_filter")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "gaussian_filter requires at least 1D input".to_string(),
        });
    }

    let sigmas = expand_sigmas(sigma, ndim);
    let orders: Vec<usize> = if order.is_empty() {
        vec![0; ndim]
    } else if order.len() >= ndim {
        order[..ndim].to_vec()
    } else {
        let mut o = order.to_vec();
        o.resize(ndim, 0);
        o
    };

    let mut result = input.clone();

    // Apply 1D Gaussian along each axis
    for axis in 0..ndim {
        if sigmas[axis] <= 0.0 {
            continue; // Skip axes with sigma <= 0
        }

        let kernel = gaussian_kernel_1d(client, sigmas[axis], orders[axis], truncate, dtype)?;
        result = convolve_along_axis(client, &result, &kernel, axis, mode)?;
    }

    Ok(result)
}

/// Generic uniform (box) filter implementation.
pub fn uniform_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    size: &[usize],
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ScalarOps<R> + ShapeOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let dtype = input.dtype();
    validate_signal_dtype(dtype, "uniform_filter")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "uniform_filter requires at least 1D input".to_string(),
        });
    }

    let sizes = expand_sizes(size, ndim);
    let mut result = input.clone();

    for (axis, &size_val) in sizes.iter().enumerate() {
        if size_val <= 1 {
            continue;
        }
        let kernel = uniform_kernel_1d(client, size_val, dtype)?;
        result = convolve_along_axis(client, &result, &kernel, axis, mode)?;
    }

    Ok(result)
}

/// Generic minimum filter implementation.
///
/// Uses separable approach: apply 1D minimum along each axis.
/// For each axis, uses padding + minimum to compute running minimum.
pub fn minimum_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    size: &[usize],
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + ReduceOps<R> + SortingOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let dtype = input.dtype();
    validate_signal_dtype(dtype, "minimum_filter")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "minimum_filter requires at least 1D input".to_string(),
        });
    }

    let sizes = expand_sizes(size, ndim);
    let mut result = input.clone();

    for (axis, &size_val) in sizes.iter().enumerate() {
        if size_val <= 1 {
            continue;
        }
        result = rank_filter_axis_impl(client, &result, axis, size_val, mode, RankOp::Min)?;
    }

    Ok(result)
}

/// Generic maximum filter implementation.
pub fn maximum_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    size: &[usize],
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + ReduceOps<R> + SortingOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let dtype = input.dtype();
    validate_signal_dtype(dtype, "maximum_filter")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "maximum_filter requires at least 1D input".to_string(),
        });
    }

    let sizes = expand_sizes(size, ndim);
    let mut result = input.clone();

    for (axis, &size_val) in sizes.iter().enumerate() {
        if size_val <= 1 {
            continue;
        }
        result = rank_filter_axis_impl(client, &result, axis, size_val, mode, RankOp::Max)?;
    }

    Ok(result)
}

/// Generic percentile filter implementation.
pub fn percentile_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    percentile: f64,
    size: &[usize],
    mode: BoundaryMode,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + ReduceOps<R> + SortingOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    if !(0.0..=100.0).contains(&percentile) {
        return Err(Error::InvalidArgument {
            arg: "percentile",
            reason: format!("Percentile must be between 0 and 100, got {percentile}"),
        });
    }

    let dtype = input.dtype();
    validate_signal_dtype(dtype, "percentile_filter")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "percentile_filter requires at least 1D input".to_string(),
        });
    }

    // For percentile == 0 or 100, delegate to min/max
    if (percentile - 0.0).abs() < 1e-10 {
        return minimum_filter_impl(client, input, size, mode);
    }
    if (percentile - 100.0).abs() < 1e-10 {
        return maximum_filter_impl(client, input, size, mode);
    }

    // General percentile: apply separable rank filter along each axis
    let sizes = expand_sizes(size, ndim);
    let mut result = input.clone();

    for (axis, &size_val) in sizes.iter().enumerate() {
        if size_val <= 1 {
            continue;
        }
        result = rank_filter_axis_impl(
            client,
            &result,
            axis,
            size_val,
            mode,
            RankOp::Percentile(percentile),
        )?;
    }

    Ok(result)
}

/// What rank operation to perform.
enum RankOp {
    Min,
    Max,
    Percentile(f64),
}

/// Apply a rank filter along a single axis using padding + reduce.
///
/// For Min/Max, this is efficient: pad, then take running min/max via
/// sequential narrow + minimum/maximum operations.
///
/// For Percentile, we gather window elements, sort, and select.
fn rank_filter_axis_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    axis: usize,
    window_size: usize,
    mode: BoundaryMode,
    op: RankOp,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + ReduceOps<R> + SortingOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let half = window_size / 2;

    // Pad along the axis
    let padded = pad_axis_impl(client, input, axis as isize, half, half, mode)?;

    // Collect shifted versions and reduce
    let original_len = input.shape()[axis];

    match op {
        RankOp::Min => {
            // Start with the first window position, iteratively take minimum
            let mut result = padded.narrow(axis as isize, 0, original_len)?;
            for i in 1..window_size {
                let shifted = padded.narrow(axis as isize, i, original_len)?;
                result = client.minimum(&result, &shifted)?;
            }
            Ok(result)
        }
        RankOp::Max => {
            let mut result = padded.narrow(axis as isize, 0, original_len)?;
            for i in 1..window_size {
                let shifted = padded.narrow(axis as isize, i, original_len)?;
                result = client.maximum(&result, &shifted)?;
            }
            Ok(result)
        }
        RankOp::Percentile(pct) => {
            // Stack all window positions, sort along window dim, pick percentile index
            let mut windows: Vec<Tensor<R>> = Vec::with_capacity(window_size);
            for i in 0..window_size {
                windows.push(padded.narrow(axis as isize, i, original_len)?);
            }
            let refs: Vec<&Tensor<R>> = windows.iter().collect();

            // Stack along a new last dimension
            let stacked = client.stack(&refs, input.ndim() as isize)?;

            // Sort along the last dimension (the window dimension)
            let window_dim = input.ndim() as isize; // new dim added by stack
            let sorted = client.sort(&stacked, window_dim, false)?;

            // Pick element at percentile position
            let idx = ((pct / 100.0) * (window_size - 1) as f64).round() as usize;
            let idx = idx.min(window_size - 1);

            let selected = sorted.narrow(window_dim, idx, 1)?;
            let squeezed = selected.squeeze(Some(window_dim));
            Ok(squeezed.contiguous()?)
        }
    }
}
