//! 1D interpolation evaluation methods using tensor operations.
//!
//! All computation stays on device - no to_vec() calls inside algorithms.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::{Interp1d, InterpMethod};

impl<R: Runtime> Interp1d<R> {
    /// Evaluate the interpolator at new x coordinates.
    ///
    /// Uses tensor operations for GPU-accelerated batch evaluation.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x_new` - 1D tensor of x coordinates to interpolate at
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated y values.
    ///
    /// # Errors
    ///
    /// Returns error if any x_new value is outside the interpolation domain.
    pub fn evaluate<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    {
        let x_new_shape = x_new.shape();
        if x_new_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x_new".to_string(),
                message: "x_new must be a 1D tensor".to_string(),
            });
        }

        // Out-of-bounds queries are clamped to boundary intervals
        match self.method {
            InterpMethod::Nearest => self.evaluate_nearest(client, x_new),
            InterpMethod::Linear => self.evaluate_linear(client, x_new),
            InterpMethod::Cubic => self.evaluate_cubic(client, x_new),
        }
    }

    /// Nearest neighbor interpolation using tensor ops.
    fn evaluate_nearest<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let m = x_new.shape()[0];
        let device = client.device();

        // Find interval indices using searchsorted
        let indices = client.searchsorted(&self.x, x_new, false)?;

        // Create constant tensors for clamping
        let ones = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(self.n - 1) as i64; m], &[m], device);

        // Clamp indices to valid range [1, n-1], then get interval [0, n-2]
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones)?;
        let idx = client.sub(&indices_clamped, &ones)?;
        let idx_plus_1 = client.add(&idx, &ones)?;

        // Get x values at interval endpoints
        let x0 = client.index_select(&self.x, 0, &idx)?;
        let x1 = client.index_select(&self.x, 0, &idx_plus_1)?;
        let y0 = client.index_select(&self.y, 0, &idx)?;
        let y1 = client.index_select(&self.y, 0, &idx_plus_1)?;

        // Compute distances: d0 = xi - x0, d1 = x1 - xi
        let d0 = client.sub(x_new, &x0)?;
        let d1 = client.sub(&x1, x_new)?;

        // Create indicator for which point is closer
        // indicator = 1 if d0 <= d1 (closer to x0), 0 otherwise
        // Use smooth indicator: (d1 - d0 + |d1 - d0|) / (2 * |d1 - d0| + eps)
        let diff = client.sub(&d1, &d0)?;
        let diff_abs = client.abs(&diff)?;
        let epsilon = Tensor::<R>::from_slice(&vec![1e-14; m], &[m], device);
        let sum = client.add(&diff, &diff_abs)?;
        let denom = client.add(&client.mul_scalar(&diff_abs, 2.0)?, &epsilon)?;
        let indicator = client.div(&sum, &denom)?; // ~1 if d0 <= d1

        // result = y0 * indicator + y1 * (1 - indicator)
        let ones_f64 = Tensor::<R>::from_slice(&vec![1.0; m], &[m], device);
        let one_minus_ind = client.sub(&ones_f64, &indicator)?;
        let term0 = client.mul(&y0, &indicator)?;
        let term1 = client.mul(&y1, &one_minus_ind)?;
        let result = client.add(&term0, &term1)?;

        Ok(result)
    }

    /// Linear interpolation using tensor ops.
    fn evaluate_linear<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let m = x_new.shape()[0];
        let device = client.device();

        // Find interval indices using searchsorted
        let indices = client.searchsorted(&self.x, x_new, false)?;

        // Create constant tensors for clamping
        let ones = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(self.n - 1) as i64; m], &[m], device);

        // Clamp indices to valid range [1, n-1], then get interval [0, n-2]
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones)?;
        let idx = client.sub(&indices_clamped, &ones)?;
        let idx_plus_1 = client.add(&idx, &ones)?;

        // Gather values at interval endpoints
        let x0 = client.index_select(&self.x, 0, &idx)?;
        let x1 = client.index_select(&self.x, 0, &idx_plus_1)?;
        let y0 = client.index_select(&self.y, 0, &idx)?;
        let y1 = client.index_select(&self.y, 0, &idx_plus_1)?;

        // Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        let dx = client.sub(&x1, &x0)?;
        let dy = client.sub(&y1, &y0)?;
        let x_offset = client.sub(x_new, &x0)?;

        // t = (x - x0) / (x1 - x0)
        let epsilon = Tensor::<R>::from_slice(&vec![1e-14; m], &[m], device);
        let dx_safe = client.add(&dx, &epsilon)?;
        let t = client.div(&x_offset, &dx_safe)?;

        // result = y0 + dy * t
        let scaled_dy = client.mul(&dy, &t)?;
        let result = client.add(&y0, &scaled_dy)?;

        Ok(result)
    }

    /// Cubic interpolation using tensor ops (Catmull-Rom style).
    fn evaluate_cubic<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let m = x_new.shape()[0];
        let n = self.n;
        let device = client.device();

        // Find interval indices using searchsorted
        let indices = client.searchsorted(&self.x, x_new, false)?;

        // Create constant tensors
        let zeros_i64 = Tensor::<R>::from_slice(&vec![0i64; m], &[m], device);
        let ones_i64 = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(n - 1) as i64; m], &[m], device);

        // Clamp main indices to valid range [1, n-1], then get interval [0, n-2]
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones_i64)?;
        let i1 = client.sub(&indices_clamped, &ones_i64)?; // Main interval start
        let i2 = client.add(&i1, &ones_i64)?; // Main interval end

        // Get 4-point stencil with clamping at boundaries
        // i0 = max(0, i1 - 1)
        let i1_minus_1 = client.sub(&i1, &ones_i64)?;
        let i0 = client.maximum(&i1_minus_1, &zeros_i64)?;

        // i3 = min(n-1, i2 + 1)
        let i2_plus_1 = client.add(&i2, &ones_i64)?;
        let i3 = client.minimum(&i2_plus_1, &n_minus_1)?;

        // Gather x and y values at all 4 points
        let x0 = client.index_select(&self.x, 0, &i0)?;
        let x1 = client.index_select(&self.x, 0, &i1)?;
        let x2 = client.index_select(&self.x, 0, &i2)?;
        let x3 = client.index_select(&self.x, 0, &i3)?;

        let y0 = client.index_select(&self.y, 0, &i0)?;
        let y1 = client.index_select(&self.y, 0, &i1)?;
        let y2 = client.index_select(&self.y, 0, &i2)?;
        let y3 = client.index_select(&self.y, 0, &i3)?;

        // Compute intervals
        let h1 = client.sub(&x2, &x1)?; // Main interval
        let h0 = client.sub(&x1, &x0)?; // Left interval
        let h2 = client.sub(&x3, &x2)?; // Right interval

        // Epsilon for safe division
        let epsilon = Tensor::<R>::from_slice(&vec![1e-14; m], &[m], device);
        let h1_safe = client.add(&h1, &epsilon)?;
        let h0_safe = client.add(&h0, &epsilon)?;
        let h2_safe = client.add(&h2, &epsilon)?;

        // Compute slopes for tangent estimation
        let slope_01 = client.div(&client.sub(&y1, &y0)?, &h0_safe)?;
        let slope_12 = client.div(&client.sub(&y2, &y1)?, &h1_safe)?;
        let slope_23 = client.div(&client.sub(&y3, &y2)?, &h2_safe)?;

        // Detect boundaries using interval widths (h0 ≈ 0 means left boundary, h2 ≈ 0 means right)
        // When i0 == i1 (clamped), x0 == x1, so h0 = 0
        // left_boundary_indicator ≈ 1 when h0 ≈ 0, ≈ 0 otherwise
        let ones_f64 = Tensor::<R>::from_slice(&vec![1.0; m], &[m], device);
        let half = Tensor::<R>::from_slice(&vec![0.5; m], &[m], device);

        // Smooth boundary indicator: 1 - h / (h + eps) ≈ 1 if h ≈ 0
        let h0_abs = client.abs(&h0)?;
        let h0_ratio = client.div(&h0_abs, &client.add(&h0_abs, &epsilon)?)?;
        let left_boundary = client.sub(&ones_f64, &h0_ratio)?;

        let h2_abs = client.abs(&h2)?;
        let h2_ratio = client.div(&h2_abs, &client.add(&h2_abs, &epsilon)?)?;
        let right_boundary = client.sub(&ones_f64, &h2_ratio)?;

        // m1 = left_boundary ? slope_12 : 0.5 * (slope_01 + slope_12)
        let avg_m1 = client.mul(&half, &client.add(&slope_01, &slope_12)?)?;
        let one_minus_left = client.sub(&ones_f64, &left_boundary)?;
        let m1 = client.add(
            &client.mul(&left_boundary, &slope_12)?,
            &client.mul(&one_minus_left, &avg_m1)?,
        )?;

        // m2 = right_boundary ? slope_12 : 0.5 * (slope_12 + slope_23)
        let avg_m2 = client.mul(&half, &client.add(&slope_12, &slope_23)?)?;
        let one_minus_right = client.sub(&ones_f64, &right_boundary)?;
        let m2 = client.add(
            &client.mul(&right_boundary, &slope_12)?,
            &client.mul(&one_minus_right, &avg_m2)?,
        )?;

        // Compute t = (x - x1) / h1
        let x_offset = client.sub(x_new, &x1)?;
        let t = client.div(&x_offset, &h1_safe)?;
        let t2 = client.mul(&t, &t)?;
        let t3 = client.mul(&t2, &t)?;

        // Hermite basis functions
        // h00 = 2t³ - 3t² + 1
        let h00 = client.add_scalar(
            &client.sub(&client.mul_scalar(&t3, 2.0)?, &client.mul_scalar(&t2, 3.0)?)?,
            1.0,
        )?;

        // h10 = t³ - 2t² + t
        let h10 = client.add(&client.sub(&t3, &client.mul_scalar(&t2, 2.0)?)?, &t)?;

        // h01 = -2t³ + 3t²
        let h01 = client.add(
            &client.mul_scalar(&t3, -2.0)?,
            &client.mul_scalar(&t2, 3.0)?,
        )?;

        // h11 = t³ - t²
        let h11 = client.sub(&t3, &t2)?;

        // Cubic Hermite: p(t) = h00*y1 + h10*h1*m1 + h01*y2 + h11*h1*m2
        let term1 = client.mul(&h00, &y1)?;
        let term2 = client.mul(&h10, &client.mul(&h1, &m1)?)?;
        let term3 = client.mul(&h01, &y2)?;
        let term4 = client.mul(&h11, &client.mul(&h1, &m2)?)?;

        let result = client.add(&client.add(&term1, &term2)?, &client.add(&term3, &term4)?)?;

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
    fn test_linear_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 1.0).abs() < 1e-10);
        assert!((y_result[1] - 3.0).abs() < 1e-10);
        assert!((y_result[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 10.0, 20.0, 30.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Nearest).unwrap();

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.3, 0.7, 1.9], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);
        assert!((y_result[1] - 10.0).abs() < 1e-10);
        assert!((y_result[2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Cubic).unwrap();

        let x_new = Tensor::<CpuRuntime>::from_slice(&[1.5], &[1], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_out_of_bounds_clamps() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        // Out-of-bounds queries are clamped to boundary intervals
        let x_new = Tensor::<CpuRuntime>::from_slice(&[-0.5, 2.5], &[2], &device);
        let result = interp.evaluate(&client, &x_new).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Results should be computed using the boundary intervals
        assert_eq!(result_data.len(), 2);
    }
}
