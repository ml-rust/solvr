//! N-dimensional grid interpolation evaluation methods using tensor operations.
//!
//! Uses vectorized operations for batch evaluation on device.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::{ExtrapolateMode, InterpNdMethod, RegularGridInterpolator};

impl<R: Runtime> RegularGridInterpolator<R> {
    /// Evaluate the interpolant at query points using tensor operations.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client
    /// * `xi` - Query points as 2D tensor of shape [n_points, ndim]
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated values with length n_points.
    pub fn evaluate<C>(&self, client: &C, xi: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let xi_shape = xi.shape();
        if xi_shape.len() != 2 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "xi".to_string(),
                message: format!(
                    "Query points must be 2D [n_points, ndim], got {:?}",
                    xi_shape
                ),
            });
        }

        let n_points = xi_shape[0];
        let query_ndim = xi_shape[1];

        if query_ndim != self.n_dims {
            return Err(InterpolateError::DimensionMismatch {
                expected: self.n_dims,
                actual: query_ndim,
                context: "RegularGridInterpolator::evaluate (query dimensions)".to_string(),
            });
        }

        // Out-of-bounds queries are clamped to boundary (Error mode behaves like Extrapolate)
        match self.method {
            InterpNdMethod::Nearest => self.evaluate_nearest_tensor(client, xi, n_points),
            InterpNdMethod::Linear => self.evaluate_linear_tensor(client, xi, n_points),
        }
    }

    /// Nearest neighbor interpolation using tensor operations.
    fn evaluate_nearest_tensor<C>(
        &self,
        client: &C,
        xi: &Tensor<R>,
        n_points: usize,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // All the work is done in gather_nd_nearest which computes
        // flat indices using F64 arithmetic and rounds to find nearest
        self.gather_nd_nearest(client, xi, n_points)
    }

    /// Multilinear interpolation using tensor operations.
    fn evaluate_linear_tensor<C>(
        &self,
        client: &C,
        xi: &Tensor<R>,
        n_points: usize,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let device = client.device();

        // For each dimension, find interval indices and fractions
        let mut lo_indices: Vec<Tensor<R>> = Vec::with_capacity(self.n_dims);
        let mut fracs: Vec<Tensor<R>> = Vec::with_capacity(self.n_dims);

        for d in 0..self.n_dims {
            // Extract query coordinates for this dimension
            let xi_d = self.extract_column(client, xi, d, n_points)?;

            // Find interval using searchsorted
            let indices = client.searchsorted(&self.points[d], &xi_d, false)?;

            // Clamp and compute lo index
            let ones = Tensor::<R>::from_slice(&vec![1i64; n_points], &[n_points], device);
            let n_d = self.shape[d];
            let n_d_minus_1 =
                Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

            let indices_clamped =
                client.maximum(&client.minimum(&indices, &n_d_minus_1)?, &ones)?;
            let idx_lo = client.sub(&indices_clamped, &ones)?;

            // Get grid values at lo and hi
            let x_lo = client.index_select(&self.points[d], 0, &idx_lo)?;
            let idx_hi = client.minimum(&indices_clamped, &n_d_minus_1)?;
            let x_hi = client.index_select(&self.points[d], 0, &idx_hi)?;

            // Compute fraction
            let dx = client.sub(&x_hi, &x_lo)?;
            let epsilon = Tensor::<R>::from_slice(&vec![1e-14; n_points], &[n_points], device);
            let dx_safe = client.add(&dx, &epsilon)?;
            let frac = client.div(&client.sub(&xi_d, &x_lo)?, &dx_safe)?;

            // Handle extrapolation
            let frac_clamped = match self.extrapolate {
                ExtrapolateMode::Nan => frac.clone(), // NaN handling done elsewhere
                _ => {
                    let zeros = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);
                    let ones_f = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
                    client.maximum(&client.minimum(&frac, &ones_f)?, &zeros)?
                }
            };

            lo_indices.push(idx_lo);
            fracs.push(frac_clamped);
        }

        // Multilinear interpolation: sum over 2^ndim vertices
        self.multilinear_interp(client, &lo_indices, &fracs, n_points)
    }

    /// Extract column d from 2D tensor xi [n_points, ndim] → [n_points]
    fn extract_column<C>(
        &self,
        _client: &C,
        xi: &Tensor<R>,
        d: usize,
        n_points: usize,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + RuntimeClient<R>,
    {
        // Use narrow to get column, then make contiguous before reshape
        // xi.narrow(1, d, 1) gives [n_points, 1]
        let col = xi.narrow(1, d, 1)?;
        let col_contig = col.contiguous();
        Ok(col_contig.reshape(&[n_points])?)
    }

    /// Gather values for nearest neighbor interpolation.
    fn gather_nd_nearest<C>(
        &self,
        client: &C,
        xi: &Tensor<R>,
        n_points: usize,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let device = client.device();
        let epsilon = Tensor::<R>::from_slice(&vec![1e-14; n_points], &[n_points], device);
        let half = Tensor::<R>::from_slice(&vec![0.5; n_points], &[n_points], device);
        let ones_f = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
        let zeros_f = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);

        // Compute flat indices for nearest neighbor
        // flat_idx = sum_d (nearest_idx[d] * stride[d])
        let mut flat_idx_f64 = zeros_f.clone();
        let mut stride: usize = 1;

        // Row-major: last dimension varies fastest
        for d in (0..self.n_dims).rev() {
            // Extract query coordinates for dimension d
            let xi_d = self.extract_column(client, xi, d, n_points)?;

            // Find interval
            let indices = client.searchsorted(&self.points[d], &xi_d, false)?;
            let ones = Tensor::<R>::from_slice(&vec![1i64; n_points], &[n_points], device);
            let n_d = self.shape[d];
            let n_d_minus_1 =
                Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

            let indices_clamped =
                client.maximum(&client.minimum(&indices, &n_d_minus_1)?, &ones)?;
            let idx_lo = client.sub(&indices_clamped, &ones)?;
            let idx_hi = client.minimum(&indices_clamped, &n_d_minus_1)?;

            // Get grid values
            let x_lo = client.index_select(&self.points[d], 0, &idx_lo)?;
            let x_hi = client.index_select(&self.points[d], 0, &idx_hi)?;

            // Compute fraction
            let dx = client.sub(&x_hi, &x_lo)?;
            let dx_safe = client.add(&dx, &epsilon)?;
            let frac = client.div(&client.sub(&xi_d, &x_lo)?, &dx_safe)?;

            // Nearest: round fraction to 0 or 1
            // offset = 1 if frac >= 0.5, else 0
            // Use smooth step: offset = step(frac - 0.5) where step smoothly transitions
            let frac_shifted = client.sub(&frac, &half)?;
            let frac_shifted_abs = client.abs(&frac_shifted)?;
            let sum = client.add(&frac_shifted, &frac_shifted_abs)?;
            let denom = client.add(&client.mul_scalar(&frac_shifted_abs, 2.0)?, &epsilon)?;
            let offset = client.div(&sum, &denom)?; // 0 if frac < 0.5, ~1 if frac >= 0.5

            // Convert idx_lo (I64) to F64 for arithmetic
            // We'll compute: nearest_idx_f64 = idx_lo_f64 + offset
            // where idx_lo_f64 comes from indexing math
            // Actually, we need to accumulate the flat index

            // Since idx_lo is I64 and we need F64, use the stride pattern
            // flat_idx += nearest_idx * stride
            // where nearest_idx = idx_lo + round(offset)

            // For F64 computation: idx_lo is already computed from searchsorted
            // We know: idx_lo ∈ [0, n_d-2], idx_hi = idx_lo + 1 ∈ [1, n_d-1]
            // nearest_idx = idx_lo + offset where offset ∈ {0, 1}

            // Gather idx_lo value directly for flat index
            // We need to convert I64 index to F64 contribution
            // Use: index_select from a range tensor

            // Create range tensor [0, 1, 2, ..., n_d-1] and index_select
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

        // Round flat index to I64 and gather
        // flat_idx_i64 = round(flat_idx_f64)
        // We need to convert F64 to I64 for index_select
        // Use floor(x + 0.5) = round(x)
        let half_tensor = Tensor::<R>::from_slice(&vec![0.5; n_points], &[n_points], device);
        let flat_idx_rounded = client.add(&flat_idx_f64, &half_tensor)?;

        // Convert F64 indices to I64 via searchsorted
        let total_size: usize = self.shape.iter().product();

        // Flatten values tensor
        let values_flat = self.values.reshape(&[total_size])?;

        // For each point, we need to convert F64 index to I64
        // Use searchsorted on the range to convert
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
        let flat_idx_clamped =
            client.maximum(&client.minimum(&flat_idx_i64, &max_idx)?, &zeros_i64)?;

        // Gather values
        let result = client.index_select(&values_flat, 0, &flat_idx_clamped)?;

        // Handle NaN for out-of-bounds if needed
        if matches!(self.extrapolate, ExtrapolateMode::Nan) {
            // Check bounds and set NaN where out of bounds
            // This requires checking each dimension
            let mut in_bounds = ones_f.clone();
            for d in 0..self.n_dims {
                let xi_d = self.extract_column(client, xi, d, n_points)?;
                let pts = &self.points[d];
                let n_d = self.shape[d];

                // min and max for this dimension - broadcast using index_select
                let zero_idx = Tensor::<R>::from_slice(&vec![0i64; n_points], &[n_points], device);
                let max_idx =
                    Tensor::<R>::from_slice(&vec![(n_d - 1) as i64; n_points], &[n_points], device);

                let min_tensor = client.index_select(pts, 0, &zero_idx)?;
                let max_tensor = client.index_select(pts, 0, &max_idx)?;

                // Check: min <= xi_d <= max
                // in_range_lo = (xi_d - min) >= 0
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

            // result = result * in_bounds + NaN * (1 - in_bounds)
            let nan_tensor =
                Tensor::<R>::from_slice(&vec![f64::NAN; n_points], &[n_points], device);
            let one_minus_bounds = client.sub(&ones_f, &in_bounds)?;
            let result_with_nan = client.add(
                &client.mul(&result, &in_bounds)?,
                &client.mul(&nan_tensor, &one_minus_bounds)?,
            )?;
            return Ok(result_with_nan);
        }

        Ok(result)
    }

    /// Multilinear interpolation over hypercube vertices.
    fn multilinear_interp<C>(
        &self,
        client: &C,
        lo_indices: &[Tensor<R>],
        fracs: &[Tensor<R>],
        n_points: usize,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let device = client.device();
        let n_vertices = 1 << self.n_dims;
        let total_size: usize = self.shape.iter().product();
        let epsilon = Tensor::<R>::from_slice(&vec![1e-14; n_points], &[n_points], device);
        let ones_f = Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);

        // Flatten values
        let values_flat = self.values.reshape(&[total_size])?;

        // Create F64 range for index conversion
        let range_f64_flat: Vec<f64> = (0..total_size).map(|i| i as f64 + 0.5).collect();
        let range_f64_tensor = Tensor::<R>::from_slice(&range_f64_flat, &[total_size], device);

        // Create per-dimension range tensors for I64→F64 conversion
        let mut dim_ranges: Vec<Tensor<R>> = Vec::with_capacity(self.n_dims);
        for d in 0..self.n_dims {
            let range: Vec<f64> = (0..self.shape[d]).map(|i| i as f64).collect();
            dim_ranges.push(Tensor::<R>::from_slice(&range, &[self.shape[d]], device));
        }

        // Accumulate weighted sum over all vertices
        let mut result = Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);

        for vertex in 0..n_vertices {
            // Compute flat index and weight for this vertex
            let mut flat_idx_f64 =
                Tensor::<R>::from_slice(&vec![0.0; n_points], &[n_points], device);
            let mut weight = ones_f.clone();
            let mut stride: usize = 1;

            for d in (0..self.n_dims).rev() {
                let use_hi = (vertex >> d) & 1 == 1;

                // Get idx_lo as F64
                let idx_lo_f64 = client.index_select(&dim_ranges[d], 0, &lo_indices[d])?;

                // Compute index for this vertex
                let idx_f64 = if use_hi {
                    let one_tensor =
                        Tensor::<R>::from_slice(&vec![1.0; n_points], &[n_points], device);
                    let max_idx = (self.shape[d] - 1) as f64;
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

                stride *= self.shape[d];
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
        if matches!(self.extrapolate, ExtrapolateMode::Nan) {
            // (same NaN handling as nearest neighbor - check bounds)
            // Simplified: just check if any fraction is outside [0, 1]
            let mut in_bounds = ones_f.clone();

            for frac in fracs.iter() {
                // Check 0 <= frac <= 1

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

            let nan_tensor =
                Tensor::<R>::from_slice(&vec![f64::NAN; n_points], &[n_points], device);
            let one_minus_bounds = client.sub(&ones_f, &in_bounds)?;
            return Ok(client.add(
                &client.mul(&result, &in_bounds)?,
                &client.mul(&nan_tensor, &one_minus_bounds)?,
            )?);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_1d_linear() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_2d_linear() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 1.0, 2.0], &[2, 2], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0, 30.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Nearest).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.3, 0.7, 1.4], &[3, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 10.0).abs() < 1e-10);
        assert!((result_data[1] - 20.0).abs() < 1e-10);
        assert!((result_data[2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_out_of_bounds_clamps() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        // Out-of-bounds queries are clamped to boundary
        let xi = Tensor::<CpuRuntime>::from_slice(&[-0.5, 2.5], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Results should be computed using the boundary intervals
        assert_eq!(result_data.len(), 2);
    }

    #[test]
    fn test_3d_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let values = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0],
            &[2, 2, 2],
            &device,
        );

        let interp =
            RegularGridInterpolator::new(&client, &[&x, &y, &z], &values, InterpNdMethod::Linear)
                .unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5, 0.5], &[1, 3], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.5).abs() < 1e-10);
    }
}
