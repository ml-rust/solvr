//! N-dimensional grid interpolation generic implementation.
//!
//! Uses vectorized operations for batch evaluation on device.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::traits::interpnd::{ExtrapolateMode, InterpNdMethod};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Evaluate N-dimensional interpolation at query points.
///
/// # Arguments
///
/// * `points` - Slice of 1D tensors (coordinate arrays for each dimension)
/// * `values` - N-dimensional tensor of grid values
/// * `xi` - Query points as 2D tensor of shape [n_points, ndim]
/// * `method` - Interpolation method (Nearest or Linear)
/// * `extrapolate` - How to handle out-of-bounds queries
///
/// # Returns
///
/// 1D tensor of interpolated values at query points.
pub fn interpnd_evaluate<R, C>(
    client: &C,
    points: &[&Tensor<R>],
    values: &Tensor<R>,
    xi: &Tensor<R>,
    method: InterpNdMethod,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let xi_shape = xi.shape();
    if xi_shape.len() != 2 {
        return Err(
            crate::interpolate::error::InterpolateError::InvalidParameter {
                parameter: "xi".to_string(),
                message: format!(
                    "Query points must be 2D [n_points, ndim], got {:?}",
                    xi_shape
                ),
            },
        );
    }

    let n_points = xi_shape[0];
    let query_ndim = xi_shape[1];
    let n_dims = points.len();

    if query_ndim != n_dims {
        return Err(
            crate::interpolate::error::InterpolateError::DimensionMismatch {
                expected: n_dims,
                actual: query_ndim,
                context: "interpnd_evaluate (query dimensions)".to_string(),
            },
        );
    }

    let values_shape = values.shape();
    if values_shape.len() != n_dims {
        return Err(
            crate::interpolate::error::InterpolateError::DimensionMismatch {
                expected: n_dims,
                actual: values_shape.len(),
                context: "interpnd_evaluate (values dimensions)".to_string(),
            },
        );
    }

    // Get shape information
    let mut shape = Vec::with_capacity(n_dims);
    for (d, &pts) in points.iter().enumerate() {
        let pts_shape = pts.shape();
        if pts_shape.len() != 1 {
            return Err(
                crate::interpolate::error::InterpolateError::InvalidParameter {
                    parameter: format!("points[{}]", d),
                    message: "Coordinate arrays must be 1D".to_string(),
                },
            );
        }
        let n = pts_shape[0];
        if n != values_shape[d] {
            return Err(crate::interpolate::error::InterpolateError::ShapeMismatch {
                expected: n,
                actual: values_shape[d],
                context: format!("interpnd_evaluate dimension {} (points vs values)", d),
            });
        }
        shape.push(n);
    }

    // Out-of-bounds queries are clamped to boundary (Error mode behaves like Extrapolate)
    match method {
        InterpNdMethod::Nearest => evaluate_nearest_tensor(
            client,
            points,
            values,
            xi,
            &shape,
            n_points,
            n_dims,
            extrapolate,
        ),
        InterpNdMethod::Linear => evaluate_linear_tensor(
            client,
            points,
            values,
            xi,
            &shape,
            n_points,
            n_dims,
            extrapolate,
        ),
    }
}

#[allow(clippy::too_many_arguments)] // All parameters necessary for N-D interpolation
fn evaluate_nearest_tensor<R, C>(
    client: &C,
    points: &[&Tensor<R>],
    values: &Tensor<R>,
    xi: &Tensor<R>,
    shape: &[usize],
    n_points: usize,
    n_dims: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let epsilon = Tensor::<R>::from_slice(&vec![1e-14; n_points], &[n_points], device);
    let half = Tensor::<R>::from_slice(&vec![0.5; n_points], &[n_points], device);
    let ones_f = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
    let zeros_f = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);

    // Compute flat indices for nearest neighbor
    let mut flat_idx_f64 = zeros_f.clone();
    let mut stride: usize = 1;

    // Row-major: last dimension varies fastest
    for d in (0..n_dims).rev() {
        // Extract query coordinates for dimension d
        let xi_d = extract_column(xi, d, n_points)?;

        // Find interval
        let indices = client.searchsorted(points[d], &xi_d, false)?;
        let ones = Tensor::<R>::from_slice(&vec![1i64; n_points], &[n_points], device);
        let n_d = shape[d];
        let n_d_minus_1 =
            Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

        let indices_clamped = client.maximum(&client.minimum(&indices, &n_d_minus_1)?, &ones)?;
        let idx_lo = client.sub(&indices_clamped, &ones)?;
        let idx_hi = client.minimum(&indices_clamped, &n_d_minus_1)?;

        // Get grid values
        let x_lo = client.index_select(points[d], 0, &idx_lo)?;
        let x_hi = client.index_select(points[d], 0, &idx_hi)?;

        // Compute fraction
        let dx = client.sub(&x_hi, &x_lo)?;
        let dx_safe = client.add(&dx, &epsilon)?;
        let frac = client.div(&client.sub(&xi_d, &x_lo)?, &dx_safe)?;

        // Nearest: round fraction to 0 or 1
        let frac_shifted = client.sub(&frac, &half)?;
        let frac_shifted_abs = client.abs(&frac_shifted)?;
        let sum = client.add(&frac_shifted, &frac_shifted_abs)?;
        let denom = client.add(&client.mul_scalar(&frac_shifted_abs, 2.0)?, &epsilon)?;
        let offset = client.div(&sum, &denom)?;

        // Create range tensor for index conversion
        let range_f64: Vec<f64> = (0..n_d).map(|i| i as f64).collect();
        let range_tensor = Tensor::<R>::from_slice(&range_f64, &[n_d], device);

        let idx_lo_f64 = client.index_select(&range_tensor, 0, &idx_lo)?;
        let idx_hi_f64 = client.index_select(&range_tensor, 0, &idx_hi)?;

        // nearest_idx_f64 = idx_lo_f64 * (1 - offset) + idx_hi_f64 * offset
        let one_minus_offset = client.sub(&ones_f, &offset)?;
        let nearest_idx_f64 = client.add(
            &client.mul(&idx_lo_f64, &one_minus_offset)?,
            &client.mul(&idx_hi_f64, &offset)?,
        )?;

        // Add to flat index
        let stride_tensor =
            Tensor::<R>::from_slice(&vec![stride as f64; n_points], &[n_points], device);
        let contribution = client.mul(&nearest_idx_f64, &stride_tensor)?;
        flat_idx_f64 = client.add(&flat_idx_f64, &contribution)?;

        stride *= n_d;
    }

    // Convert F64 indices to I64
    let total_size: usize = shape.iter().product();
    let values_flat = values.reshape(&[total_size])?;

    // Use searchsorted for F64 to I64 conversion
    let half_tensor = Tensor::<R>::from_slice(&vec![0.5; n_points], &[n_points], device);
    let flat_idx_rounded = client.add(&flat_idx_f64, &half_tensor)?;

    let range_f64_flat: Vec<f64> = (0..total_size).map(|i| i as f64 + 0.5).collect();
    let range_f64_tensor = Tensor::<R>::from_slice(&range_f64_flat, &[total_size], device);

    let flat_idx_i64 = client.searchsorted(&range_f64_tensor, &flat_idx_rounded, false)?;

    // Clamp to valid range
    let zeros_i64 = Tensor::<R>::from_slice(&vec![0i64; n_points], &[n_points], device);
    let max_idx = Tensor::<R>::from_slice(
        &vec![(total_size - 1) as i64; n_points],
        &[n_points],
        device,
    );
    let flat_idx_clamped = client.maximum(&client.minimum(&flat_idx_i64, &max_idx)?, &zeros_i64)?;

    // Gather values
    let result = client.index_select(&values_flat, 0, &flat_idx_clamped)?;

    // Handle NaN for out-of-bounds if needed
    if matches!(extrapolate, ExtrapolateMode::Nan) {
        let mut in_bounds = ones_f.clone();
        for d in 0..n_dims {
            let xi_d = extract_column(xi, d, n_points)?;
            let pts = points[d];
            let n_d = shape[d];

            let zero_idx = Tensor::<R>::from_slice(&vec![0i64; n_points], &[n_points], device);
            let max_idx =
                Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

            let min_tensor = client.index_select(pts, 0, &zero_idx)?;
            let max_tensor = client.index_select(pts, 0, &max_idx)?;

            let diff_lo = client.sub(&xi_d, &min_tensor)?;
            let diff_lo_abs = client.abs(&diff_lo)?;
            let in_lo = client.div(
                &client.add(&diff_lo, &diff_lo_abs)?,
                &client.add(&client.mul_scalar(&diff_lo_abs, 2.0)?, &epsilon)?,
            )?;

            let diff_hi = client.sub(&max_tensor, &xi_d)?;
            let diff_hi_abs = client.abs(&diff_hi)?;
            let in_hi = client.div(
                &client.add(&diff_hi, &diff_hi_abs)?,
                &client.add(&client.mul_scalar(&diff_hi_abs, 2.0)?, &epsilon)?,
            )?;

            in_bounds = client.mul(&in_bounds, &client.mul(&in_lo, &in_hi)?)?;
        }

        let nan_tensor = Tensor::<R>::from_slice(&vec![f64::NAN; n_points], &[n_points], device);
        let one_minus_bounds = client.sub(&ones_f, &in_bounds)?;
        return Ok(client.add(
            &client.mul(&result, &in_bounds)?,
            &client.mul(&nan_tensor, &one_minus_bounds)?,
        )?);
    }

    Ok(result)
}

#[allow(clippy::too_many_arguments)] // All parameters necessary for N-D interpolation
fn evaluate_linear_tensor<R, C>(
    client: &C,
    points: &[&Tensor<R>],
    values: &Tensor<R>,
    xi: &Tensor<R>,
    shape: &[usize],
    n_points: usize,
    n_dims: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let total_size: usize = shape.iter().product();
    let values_flat = values.reshape(&[total_size])?;
    let epsilon = Tensor::<R>::from_slice(&vec![1e-14; n_points], &[n_points], device);
    let ones_f = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);

    // For each dimension, find interval indices and fractions
    let mut lo_indices: Vec<Tensor<R>> = Vec::with_capacity(n_dims);
    let mut fracs: Vec<Tensor<R>> = Vec::with_capacity(n_dims);

    for d in 0..n_dims {
        // Extract query coordinates for this dimension
        let xi_d = extract_column(xi, d, n_points)?;

        // Find interval using searchsorted
        let indices = client.searchsorted(points[d], &xi_d, false)?;

        // Clamp and compute lo index
        let ones = Tensor::<R>::from_slice(&vec![1i64; n_points], &[n_points], device);
        let n_d = shape[d];
        let n_d_minus_1 =
            Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

        let indices_clamped = client.maximum(&client.minimum(&indices, &n_d_minus_1)?, &ones)?;
        let idx_lo = client.sub(&indices_clamped, &ones)?;

        // Get grid values at lo and hi
        let x_lo = client.index_select(points[d], 0, &idx_lo)?;
        let idx_hi = client.minimum(&indices_clamped, &n_d_minus_1)?;
        let x_hi = client.index_select(points[d], 0, &idx_hi)?;

        // Compute fraction
        let dx = client.sub(&x_hi, &x_lo)?;
        let dx_safe = client.add(&dx, &epsilon)?;
        let frac = client.div(&client.sub(&xi_d, &x_lo)?, &dx_safe)?;

        // Handle extrapolation
        let frac_clamped = match extrapolate {
            ExtrapolateMode::Nan => frac.clone(),
            _ => {
                let zeros = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);
                let ones_frac = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
                client.maximum(&client.minimum(&frac, &ones_frac)?, &zeros)?
            }
        };

        lo_indices.push(idx_lo);
        fracs.push(frac_clamped);
    }

    // Multilinear interpolation: sum over 2^ndim vertices
    let n_vertices = 1 << n_dims;

    // Create per-dimension range tensors for I64→F64 conversion
    let mut dim_ranges: Vec<Tensor<R>> = Vec::with_capacity(n_dims);
    #[allow(clippy::needless_range_loop)] // Need index to access shape[d]
    for d in 0..n_dims {
        let range: Vec<f64> = (0..shape[d]).map(|i| i as f64).collect();
        dim_ranges.push(Tensor::<R>::from_slice(&range, &[shape[d]], device));
    }

    // Create F64 range for index conversion
    let range_f64_flat: Vec<f64> = (0..total_size).map(|i| i as f64 + 0.5).collect();
    let range_f64_tensor = Tensor::<R>::from_slice(&range_f64_flat, &[total_size], device);

    // Accumulate weighted sum over all vertices
    let mut result = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);

    for vertex in 0..n_vertices {
        // Compute flat index and weight for this vertex
        let mut flat_idx_f64 = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);
        let mut weight = ones_f.clone();
        let mut stride: usize = 1;

        for d in (0..n_dims).rev() {
            let use_hi = (vertex >> d) & 1 == 1;

            // Get idx_lo as F64
            let idx_lo_f64 = client.index_select(&dim_ranges[d], 0, &lo_indices[d])?;

            // Compute index for this vertex
            let idx_f64 = if use_hi {
                let one_tensor = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
                let max_idx = (shape[d] - 1) as f64;
                let max_tensor =
                    Tensor::<R>::from_slice(&vec![max_idx; n_points], &[n_points], device);
                client.minimum(&client.add(&idx_lo_f64, &one_tensor)?, &max_tensor)?
            } else {
                idx_lo_f64
            };

            // Update weight
            weight = if use_hi {
                client.mul(&weight, &fracs[d])?
            } else {
                client.mul(&weight, &client.sub(&ones_f, &fracs[d])?)?
            };

            // Add to flat index
            let stride_tensor =
                Tensor::<R>::from_slice(&vec![stride as f64; n_points], &[n_points], device);
            flat_idx_f64 = client.add(&flat_idx_f64, &client.mul(&idx_f64, &stride_tensor)?)?;

            stride *= shape[d];
        }

        // Convert F64 index to I64 using searchsorted
        let half = Tensor::<R>::from_slice(&vec![0.5; n_points], &[n_points], device);
        let flat_idx_rounded = client.add(&flat_idx_f64, &half)?;
        let flat_idx_i64 = client.searchsorted(&range_f64_tensor, &flat_idx_rounded, false)?;

        // Clamp
        let zeros_i64 = Tensor::<R>::from_slice(&vec![0i64; n_points], &[n_points], device);
        let max_idx_i64 = Tensor::<R>::from_slice(
            &vec![(total_size - 1) as i64; n_points],
            &[n_points],
            device,
        );
        let flat_idx_clamped =
            client.maximum(&client.minimum(&flat_idx_i64, &max_idx_i64)?, &zeros_i64)?;

        // Gather and accumulate
        let values_at_vertex = client.index_select(&values_flat, 0, &flat_idx_clamped)?;
        result = client.add(&result, &client.mul(&values_at_vertex, &weight)?)?;
    }

    // Handle NaN for extrapolation
    if matches!(extrapolate, ExtrapolateMode::Nan) {
        let mut in_bounds = ones_f.clone();

        for frac in fracs.iter() {
            let in_lo = client.div(
                &client.add(frac, &client.abs(frac)?)?,
                &client.add(&client.mul_scalar(&client.abs(frac)?, 2.0)?, &epsilon)?,
            )?;

            let diff_hi = client.sub(&ones_f, frac)?;
            let in_hi = client.div(
                &client.add(&diff_hi, &client.abs(&diff_hi)?)?,
                &client.add(&client.mul_scalar(&client.abs(&diff_hi)?, 2.0)?, &epsilon)?,
            )?;

            in_bounds = client.mul(&in_bounds, &client.mul(&in_lo, &in_hi)?)?;
        }

        let nan_tensor = Tensor::<R>::from_slice(&vec![f64::NAN; n_points], &[n_points], device);
        let one_minus_bounds = client.sub(&ones_f, &in_bounds)?;
        return Ok(client.add(
            &client.mul(&result, &in_bounds)?,
            &client.mul(&nan_tensor, &one_minus_bounds)?,
        )?);
    }

    Ok(result)
}

fn extract_column<R: Runtime<DType = DType>>(
    xi: &Tensor<R>,
    d: usize,
    n_points: usize,
) -> Result<Tensor<R>, crate::interpolate::error::InterpolateError> {
    // Use narrow to get column, then make contiguous before reshape
    let col = xi.narrow(1, d, 1)?;
    let col_contig = col.contiguous();
    Ok(col_contig.reshape(&[n_points])?)
}
