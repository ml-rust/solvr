//! Generic edge detection implementations.
//!
//! All edge detection operations are compositions of separable convolutions:
//! - Sobel/Prewitt: derivative kernel along target axis, smoothing kernel along others
//! - Laplace: sum of second-difference kernels along each axis
//! - LoG: Gaussian smooth then Laplace
use crate::DType;

use crate::signal::impl_generic::boundary::pad_axis_impl;
use crate::signal::impl_generic::kernels::{edge_kernel_1d, laplace_kernel_1d};
use crate::signal::traits::nd_filters::BoundaryMode;
use crate::signal::validate_signal_dtype;
use numr::error::{Error, Result};
use numr::ops::{ConvOps, PaddingMode, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Apply a separable edge filter (Sobel or Prewitt) along a given axis.
///
/// For each axis of the input:
/// - target axis: apply derivative kernel [-1, 0, 1]
/// - other axes: apply smoothing kernel [1, 2, 1] (Sobel) or [1, 1, 1] (Prewitt)
fn separable_edge_filter_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    axis: usize,
    kind: &str,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let dtype = input.dtype();

    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!("{kind} requires at least 2D input, got {ndim}D"),
        });
    }

    if axis >= ndim {
        return Err(Error::InvalidArgument {
            arg: "axis",
            reason: format!("Axis {axis} out of range for {ndim}D input"),
        });
    }

    let mut result = input.clone();

    // Apply kernels along each axis
    for ax in 0..ndim {
        let kernel = if ax == axis {
            // Derivative direction
            edge_kernel_1d(client, kind, true, dtype)?
        } else {
            // Smoothing direction
            edge_kernel_1d(client, kind, false, dtype)?
        };

        result = convolve_along_axis_simple(client, &result, &kernel, ax)?;
    }

    Ok(result)
}

/// Simple 1D convolution along an axis with Reflect boundary (used by edge detection).
fn convolve_along_axis_simple<R, C>(
    client: &C,
    input: &Tensor<R>,
    kernel_1d: &Tensor<R>,
    axis: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let shape = input.shape().to_vec();
    let ndim = shape.len();
    let kernel_len = kernel_1d.shape()[0];
    let pad_size = kernel_len / 2;

    // Pad with reflect
    let padded = pad_axis_impl(
        client,
        input,
        axis as isize,
        pad_size,
        pad_size,
        BoundaryMode::Reflect,
    )?;

    let padded_shape = padded.shape().to_vec();
    let axis_len = padded_shape[axis];
    let batch_size: usize = padded_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .product();

    // Permute axis to last
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.remove(axis);
    perm.push(axis);

    let permuted = padded.permute(&perm)?;
    let permuted_contig = permuted.contiguous()?;
    let reshaped = permuted_contig.reshape(&[batch_size, 1, axis_len])?;

    let kernel_3d = kernel_1d.reshape(&[1, 1, kernel_len])?;
    let conv_result = client.conv1d(&reshaped, &kernel_3d, None, 1, PaddingMode::Valid, 1, 1)?;

    let output_axis_len = shape[axis];
    let mut permuted_shape: Vec<usize> = perm[..ndim - 1].iter().map(|&i| shape[i]).collect();
    permuted_shape.push(output_axis_len);

    let reshaped_back = conv_result.reshape(&permuted_shape)?;

    let mut inv_perm = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    let permuted = reshaped_back.permute(&inv_perm)?;
    permuted.contiguous()
}

/// Generic Sobel filter implementation.
pub fn sobel_impl<R, C>(client: &C, input: &Tensor<R>, axis: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    validate_signal_dtype(input.dtype(), "sobel")?;
    separable_edge_filter_impl(client, input, axis, "sobel")
}

/// Generic Prewitt filter implementation.
pub fn prewitt_impl<R, C>(client: &C, input: &Tensor<R>, axis: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    validate_signal_dtype(input.dtype(), "prewitt")?;
    separable_edge_filter_impl(client, input, axis, "prewitt")
}

/// Generic Laplacian filter implementation.
///
/// Computes sum of second-difference [1, -2, 1] convolutions along each axis.
pub fn laplace_impl<R, C>(client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ScalarOps<R> + ShapeOps<R> + TensorOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    validate_signal_dtype(input.dtype(), "laplace")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "laplace requires at least 1D input".to_string(),
        });
    }

    let dtype = input.dtype();

    // Sum of [1, -2, 1] convolution along each axis
    let kernel = laplace_kernel_1d(client, dtype)?;

    let first = convolve_along_axis_simple(client, input, &kernel, 0)?;
    let mut result = first;

    for axis in 1..ndim {
        let component = convolve_along_axis_simple(client, input, &kernel, axis)?;
        result = client.add(&result, &component)?;
    }

    Ok(result)
}

/// Generic Gaussian-Laplacian (LoG) implementation.
///
/// Applies Gaussian smoothing then Laplacian.
pub fn gaussian_laplace_impl<R, C>(client: &C, input: &Tensor<R>, sigma: f64) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    validate_signal_dtype(input.dtype(), "gaussian_laplace")?;

    let ndim = input.ndim();
    let sigmas = vec![sigma; ndim];
    let orders = vec![0usize; ndim];

    // Import the gaussian_filter_impl function
    use super::nd_filters::gaussian_filter_impl;

    // Gaussian smooth
    let smoothed =
        gaussian_filter_impl(client, input, &sigmas, &orders, BoundaryMode::Reflect, 4.0)?;

    // Laplacian
    laplace_impl(client, &smoothed)
}

/// Generic Gaussian gradient magnitude implementation.
///
/// Smooths, computes gradient along each axis, returns sqrt(sum of squares).
pub fn gaussian_gradient_magnitude_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    sigma: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    validate_signal_dtype(input.dtype(), "gaussian_gradient_magnitude")?;

    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "gaussian_gradient_magnitude requires at least 1D input".to_string(),
        });
    }

    // Import the gaussian_filter_impl function
    use super::nd_filters::gaussian_filter_impl;

    // For each axis: compute Gaussian derivative (order=1) along that axis,
    // Gaussian smooth (order=0) along all other axes
    let mut sum_sq: Option<Tensor<R>> = None;

    for axis in 0..ndim {
        // Build per-axis sigma and order vectors
        let sigmas = vec![sigma; ndim];
        let mut orders = vec![0usize; ndim];
        orders[axis] = 1;

        let grad =
            gaussian_filter_impl(client, input, &sigmas, &orders, BoundaryMode::Reflect, 4.0)?;
        let grad_sq = client.mul(&grad, &grad)?;

        sum_sq = Some(match sum_sq {
            Some(acc) => client.add(&acc, &grad_sq)?,
            None => grad_sq,
        });
    }

    // sum_sq is guaranteed Some since ndim >= 1 (validated above)
    match sum_sq {
        Some(sq) => client.sqrt(&sq),
        None => Err(Error::InvalidArgument {
            arg: "input",
            reason: "gaussian_gradient_magnitude requires at least 1D input".to_string(),
        }),
    }
}
