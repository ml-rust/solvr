//! Generic geometric transform implementations.
//!
//! All transforms generate output coordinates, apply coordinate mapping,
//! then interpolate values from the input using map_coordinates.
use crate::DType;

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::geometric::InterpolationOrder;
use numr::ops::{
    CompareOps, ConditionalOps, MatmulOps, MeshgridIndexing, ReduceOps, ScalarOps, ShapeOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic map_coordinates implementation.
///
/// For nearest-neighbor: round coordinates, clamp to bounds, gather.
/// For linear: floor/ceil coordinates, compute weights, interpolate.
pub fn map_coordinates_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    coordinates: &Tensor<R>,
    order: InterpolationOrder,
    _cval: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let coord_shape = coordinates.shape().to_vec();

    if coord_shape[0] != ndim {
        return Err(InterpolateError::DimensionMismatch {
            expected: ndim,
            actual: coord_shape[0],
            context: "map_coordinates: coordinates first dim must match input ndim".to_string(),
        });
    }

    let output_shape = &coord_shape[1..];
    let output_total: usize = output_shape.iter().product();
    let input_shape = input.shape().to_vec();
    let dtype = input.dtype();

    // Flatten input for easier indexing
    let input_flat = input
        .contiguous()
        .reshape(&[input_shape.iter().product::<usize>()])?;

    // Extract and flatten coordinate arrays for each dimension
    let mut coord_arrays: Vec<Tensor<R>> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let c = coordinates.narrow(0, d, 1)?;
        let c_squeezed = c.squeeze(Some(0)).contiguous();
        let c_flat = c_squeezed.reshape(&[output_total])?;
        coord_arrays.push(c_flat);
    }

    match order {
        InterpolationOrder::Nearest => {
            // Round coordinates and clamp to bounds
            let mut flat_idx = client.fill(&[output_total], 0.0, dtype)?;
            let mut stride = 1.0f64;

            for d in (0..ndim).rev() {
                let rounded = client.round(&coord_arrays[d])?;
                let clamped = client.clamp(&rounded, 0.0, (input_shape[d] - 1) as f64)?;
                let contribution = client.mul_scalar(&clamped, stride)?;
                flat_idx = client.add(&flat_idx, &contribution)?;
                stride *= input_shape[d] as f64;
            }

            // Convert to integer indices and gather
            let idx_flat = flat_idx.reshape(&[output_total])?;
            let idx_int = client.cast(&idx_flat, numr::dtype::DType::I64)?;
            let result = client.index_select(&input_flat, 0, &idx_int)?;

            result.reshape(output_shape).map_err(|e| e.into())
        }
        InterpolationOrder::Linear => {
            // N-dimensional linear interpolation
            let num_corners = 1usize << ndim; // 2^ndim corners

            let mut floors: Vec<Tensor<R>> = Vec::with_capacity(ndim);
            let mut weights: Vec<Tensor<R>> = Vec::with_capacity(ndim);

            // Precompute floor values and weights for each dimension
            for d in 0..ndim {
                let f = client.floor(&coord_arrays[d])?;
                let w = client.sub(&coord_arrays[d], &f)?; // fractional part
                let f_clamped = client.clamp(&f, 0.0, (input_shape[d] - 1) as f64)?;
                floors.push(f_clamped);
                weights.push(w);
            }

            // Accumulate interpolated result
            let mut result = client.fill(&[output_total], 0.0, dtype)?;

            // Iterate over all 2^ndim corners
            for corner in 0..num_corners {
                let mut flat_idx = client.fill(&[output_total], 0.0, dtype)?;
                let mut corner_weight = client.fill(&[output_total], 1.0, dtype)?;
                let mut stride = 1.0f64;

                for d in (0..ndim).rev() {
                    let use_ceil = (corner >> d) & 1 == 1;

                    let coord = if use_ceil {
                        let ceil = client.add_scalar(&floors[d], 1.0)?;
                        client.clamp(&ceil, 0.0, (input_shape[d] - 1) as f64)?
                    } else {
                        floors[d].clone()
                    };

                    let w = if use_ceil {
                        weights[d].clone() // weight for ceil
                    } else {
                        let one = client.fill(&[output_total], 1.0, dtype)?;
                        client.sub(&one, &weights[d])? // 1 - weight for floor
                    };

                    corner_weight = client.mul(&corner_weight, &w)?;
                    let contribution = client.mul_scalar(&coord, stride)?;
                    flat_idx = client.add(&flat_idx, &contribution)?;
                    stride *= input_shape[d] as f64;
                }

                // Gather values at this corner
                let idx_clamped = flat_idx.reshape(&[output_total])?;
                let idx_int = client.cast(&idx_clamped, numr::dtype::DType::I64)?;
                let values = client.index_select(&input_flat, 0, &idx_int)?;
                let weighted = client.mul(&values, &corner_weight)?;
                result = client.add(&result, &weighted)?;
            }

            result.reshape(output_shape).map_err(|e| e.into())
        }
    }
}

/// Generic affine_transform implementation.
pub fn affine_transform_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    matrix: &Tensor<R>,
    offset: &Tensor<R>,
    output_shape: Option<&[usize]>,
    order: InterpolationOrder,
    cval: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let dtype = input.dtype();
    let out_shape = output_shape.unwrap_or(input.shape()).to_vec();

    // Generate output coordinate grid
    let coordinates = generate_coordinate_grid(client, &out_shape, dtype)?;
    // coordinates shape: [ndim, total_output_points]

    let total: usize = out_shape.iter().product();

    // Reshape coordinates to [ndim, total]
    let coords_flat = coordinates.reshape(&[ndim, total])?;

    // Apply affine: input_coords = matrix @ output_coords + offset
    let mapped = client.matmul(matrix, &coords_flat)?; // [ndim, total]

    // Add offset (broadcast)
    let offset_col = offset.reshape(&[ndim, 1])?;
    let mapped_with_offset = client.add(&mapped, &offset_col)?;

    // Reshape to [ndim, ...output_shape]
    let mut coord_shape = vec![ndim];
    coord_shape.extend_from_slice(&out_shape);
    let final_coords = mapped_with_offset.reshape(&coord_shape)?;

    map_coordinates_impl(client, input, &final_coords, order, cval)
}

/// Generate a coordinate grid for the given shape (fully on-device).
///
/// Uses arange + meshgrid to build coordinates without CPU scalar loops.
/// Returns tensor of shape [ndim, total] where each row contains
/// the coordinates for that dimension.
fn generate_coordinate_grid<R, C>(
    client: &C,
    shape: &[usize],
    dtype: numr::dtype::DType,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + UtilityOps<R> + ShapeOps<R> + TypeConversionOps<R> + RuntimeClient<R>,
{
    let total: usize = shape.iter().product();

    // Create 1D coordinate vectors on-device via arange
    let axes: Vec<Tensor<R>> = shape
        .iter()
        .map(|&s| client.arange(0.0, s as f64, 1.0, dtype))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // meshgrid expands each 1D vector into an N-D grid
    let axis_refs: Vec<&Tensor<R>> = axes.iter().collect();
    let grids = client.meshgrid(&axis_refs, MeshgridIndexing::Ij)?;

    // Flatten each grid to [total] and stack into [ndim, total]
    let flat_grids: Vec<Tensor<R>> = grids
        .into_iter()
        .map(|g| g.reshape(&[total]).map_err(InterpolateError::from))
        .collect::<InterpolateResult<Vec<_>>>()?;

    let flat_refs: Vec<&Tensor<R>> = flat_grids.iter().collect();
    client.stack(&flat_refs, 0).map_err(|e| e.into())
}

/// Generic zoom implementation.
pub fn zoom_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    zoom: &[f64],
    order: InterpolationOrder,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let dtype = input.dtype();
    let device = client.device();

    if zoom.len() != ndim {
        return Err(InterpolateError::InvalidParameter {
            parameter: "zoom".to_string(),
            message: format!(
                "zoom length ({}) must match input ndim ({})",
                zoom.len(),
                ndim
            ),
        });
    }

    // Compute output shape
    let output_shape: Vec<usize> = input
        .shape()
        .iter()
        .zip(zoom.iter())
        .map(|(&s, &z)| ((s as f64) * z).round() as usize)
        .collect();

    // Build diagonal matrix (inverse zoom)
    let mut matrix_data = vec![0.0f64; ndim * ndim];
    for d in 0..ndim {
        matrix_data[d * ndim + d] = 1.0 / zoom[d];
    }
    let matrix = Tensor::from_slice(&matrix_data, &[ndim, ndim], device);
    let matrix_typed = client.cast(&matrix, dtype)?;

    let offset_data = vec![0.0f64; ndim];
    let offset = Tensor::from_slice(&offset_data, &[ndim], device);
    let offset_typed = client.cast(&offset, dtype)?;

    affine_transform_impl(
        client,
        input,
        &matrix_typed,
        &offset_typed,
        Some(&output_shape),
        order,
        0.0,
    )
}

/// Generic rotate implementation.
pub fn rotate_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    angle: f64,
    axes: (usize, usize),
    reshape: bool,
    order: InterpolationOrder,
    cval: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let dtype = input.dtype();
    let device = client.device();

    if ndim < 2 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "input".to_string(),
            message: "rotate requires at least 2D input".to_string(),
        });
    }

    let (ax0, ax1) = axes;
    if ax0 >= ndim || ax1 >= ndim || ax0 == ax1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "axes".to_string(),
            message: format!("Invalid axes ({}, {}) for {}D input", ax0, ax1, ndim),
        });
    }

    let angle_rad = angle.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Build rotation matrix (identity with rotation in the plane of axes)
    let mut matrix_data = vec![0.0f64; ndim * ndim];
    for d in 0..ndim {
        matrix_data[d * ndim + d] = 1.0;
    }
    matrix_data[ax0 * ndim + ax0] = cos_a;
    matrix_data[ax0 * ndim + ax1] = sin_a;
    matrix_data[ax1 * ndim + ax0] = -sin_a;
    matrix_data[ax1 * ndim + ax1] = cos_a;

    let shape = input.shape().to_vec();

    // Compute output shape and offset
    let output_shape;
    let offset_data;

    if reshape {
        // Compute bounding box of rotated image
        let h = shape[ax0] as f64;
        let w = shape[ax1] as f64;
        let corners = [(0.0, 0.0), (h, 0.0), (0.0, w), (h, w)];
        let mut min0 = f64::MAX;
        let mut max0 = f64::MIN;
        let mut min1 = f64::MAX;
        let mut max1 = f64::MIN;
        for &(r, c) in &corners {
            let nr = cos_a * r - sin_a * c;
            let nc = sin_a * r + cos_a * c;
            min0 = min0.min(nr);
            max0 = max0.max(nr);
            min1 = min1.min(nc);
            max1 = max1.max(nc);
        }
        let new_h = (max0 - min0).ceil() as usize;
        let new_w = (max1 - min1).ceil() as usize;

        let mut os = shape.clone();
        os[ax0] = new_h;
        os[ax1] = new_w;
        output_shape = os;

        // Offset to center the rotated image
        let mut od = vec![0.0f64; ndim];
        let center_in_0 = (shape[ax0] as f64 - 1.0) / 2.0;
        let center_in_1 = (shape[ax1] as f64 - 1.0) / 2.0;
        let center_out_0 = (new_h as f64 - 1.0) / 2.0;
        let center_out_1 = (new_w as f64 - 1.0) / 2.0;
        od[ax0] = center_in_0 - cos_a * center_out_0 - sin_a * center_out_1;
        od[ax1] = center_in_1 + sin_a * center_out_0 - cos_a * center_out_1;
        offset_data = od;
    } else {
        output_shape = shape.clone();
        let center_0 = (shape[ax0] as f64 - 1.0) / 2.0;
        let center_1 = (shape[ax1] as f64 - 1.0) / 2.0;
        let mut od = vec![0.0f64; ndim];
        od[ax0] = center_0 - cos_a * center_0 - sin_a * center_1;
        od[ax1] = center_1 + sin_a * center_0 - cos_a * center_1;
        offset_data = od;
    };

    let matrix = Tensor::from_slice(&matrix_data, &[ndim, ndim], device);
    let matrix_typed = client.cast(&matrix, dtype)?;
    let offset = Tensor::from_slice(&offset_data, &[ndim], device);
    let offset_typed = client.cast(&offset, dtype)?;

    affine_transform_impl(
        client,
        input,
        &matrix_typed,
        &offset_typed,
        Some(&output_shape),
        order,
        cval,
    )
}

/// Generic shift implementation.
pub fn shift_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    shift: &[f64],
    order: InterpolationOrder,
    cval: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let dtype = input.dtype();
    let device = client.device();

    if shift.len() != ndim {
        return Err(InterpolateError::InvalidParameter {
            parameter: "shift".to_string(),
            message: format!(
                "shift length ({}) must match input ndim ({})",
                shift.len(),
                ndim
            ),
        });
    }

    // Identity matrix
    let mut matrix_data = vec![0.0f64; ndim * ndim];
    for d in 0..ndim {
        matrix_data[d * ndim + d] = 1.0;
    }

    // Negative shift as offset (mapping output coords to input coords)
    let offset_data: Vec<f64> = shift.iter().map(|&s| -s).collect();

    let matrix = Tensor::from_slice(&matrix_data, &[ndim, ndim], device);
    let matrix_typed = client.cast(&matrix, dtype)?;
    let offset = Tensor::from_slice(&offset_data, &[ndim], device);
    let offset_typed = client.cast(&offset, dtype)?;

    affine_transform_impl(
        client,
        input,
        &matrix_typed,
        &offset_typed,
        None,
        order,
        cval,
    )
}
