//! Cubic spline evaluation methods using tensor operations.
//!
//! All evaluation uses tensor ops for GPU-accelerated batch computation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::CubicSpline;

impl<R: Runtime> CubicSpline<R> {
    /// Evaluate the spline at new x coordinates using tensor operations.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x_new` - 1D tensor of x coordinates to evaluate at
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated y values.
    pub fn evaluate<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let x_new_shape = x_new.shape();
        if x_new_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x_new".to_string(),
                message: "x_new must be a 1D tensor".to_string(),
            });
        }

        let m = x_new_shape[0];
        let n = self.n;
        let device = client.device();

        // Find interval indices using searchsorted
        let indices = client.searchsorted(&self.x, x_new, false)?;

        // Clamp indices to valid interval range [1, n-1] -> [0, n-2]
        // Out-of-bounds queries are clamped to boundary intervals
        let ones = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(n - 1) as i64; m], &[m], device);
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones)?;
        let idx = client.sub(&indices_clamped, &ones)?;

        // Gather x values at interval starts
        let x_i = client.index_select(&self.x, 0, &idx)?;

        // Gather coefficients: a[idx], b[idx], c[idx], d[idx]
        let a_i = client.index_select(&self.a, 0, &idx)?;
        let b_i = client.index_select(&self.b, 0, &idx)?;
        let c_i = client.index_select(&self.c, 0, &idx)?;
        let d_i = client.index_select(&self.d, 0, &idx)?;

        // Compute dx = x_new - x_i
        let dx = client.sub(x_new, &x_i)?;

        // Evaluate polynomial: a + dx * (b + dx * (c + dx * d))
        // Using Horner's method for stability
        let term_d = client.mul(&dx, &d_i)?;
        let term_cd = client.add(&c_i, &term_d)?;
        let term_bcd = client.add(&b_i, &client.mul(&dx, &term_cd)?)?;
        let result = client.add(&a_i, &client.mul(&dx, &term_bcd)?)?;

        Ok(result)
    }

    /// Evaluate the first derivative of the spline using tensor operations.
    /// S'(x) = b + 2*c*dx + 3*d*dx^2
    pub fn derivative<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let x_new_shape = x_new.shape();
        if x_new_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x_new".to_string(),
                message: "x_new must be a 1D tensor".to_string(),
            });
        }

        let m = x_new_shape[0];
        let n = self.n;
        let device = client.device();

        // Find interval indices (out-of-bounds queries clamped to boundary)
        let indices = client.searchsorted(&self.x, x_new, false)?;
        let ones = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(n - 1) as i64; m], &[m], device);
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones)?;
        let idx = client.sub(&indices_clamped, &ones)?;

        // Gather values
        let x_i = client.index_select(&self.x, 0, &idx)?;
        let b_i = client.index_select(&self.b, 0, &idx)?;
        let c_i = client.index_select(&self.c, 0, &idx)?;
        let d_i = client.index_select(&self.d, 0, &idx)?;

        // dx = x_new - x_i
        let dx = client.sub(x_new, &x_i)?;
        let dx2 = client.mul(&dx, &dx)?;

        // S'(x) = b + 2*c*dx + 3*d*dx^2
        let term_c = client.mul_scalar(&client.mul(&c_i, &dx)?, 2.0)?;
        let term_d = client.mul_scalar(&client.mul(&d_i, &dx2)?, 3.0)?;
        let result = client.add(&b_i, &client.add(&term_c, &term_d)?)?;

        Ok(result)
    }

    /// Evaluate the second derivative of the spline using tensor operations.
    /// S''(x) = 2*c + 6*d*dx
    pub fn second_derivative<C>(
        &self,
        client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        let x_new_shape = x_new.shape();
        if x_new_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x_new".to_string(),
                message: "x_new must be a 1D tensor".to_string(),
            });
        }

        let m = x_new_shape[0];
        let n = self.n;
        let device = client.device();

        // Find interval indices (out-of-bounds queries clamped to boundary)
        let indices = client.searchsorted(&self.x, x_new, false)?;
        let ones = Tensor::<R>::from_slice(&vec![1i64; m], &[m], device);
        let n_minus_1 = Tensor::<R>::from_slice(&vec![(n - 1) as i64; m], &[m], device);
        let indices_clamped = client.maximum(&client.minimum(&indices, &n_minus_1)?, &ones)?;
        let idx = client.sub(&indices_clamped, &ones)?;

        // Gather values
        let x_i = client.index_select(&self.x, 0, &idx)?;
        let c_i = client.index_select(&self.c, 0, &idx)?;
        let d_i = client.index_select(&self.d, 0, &idx)?;

        // dx = x_new - x_i
        let dx = client.sub(x_new, &x_i)?;

        // S''(x) = 2*c + 6*d*dx
        let term_c = client.mul_scalar(&c_i, 2.0)?;
        let term_d = client.mul_scalar(&client.mul(&d_i, &dx)?, 6.0)?;
        let result = client.add(&term_c, &term_d)?;

        Ok(result)
    }

    /// Compute the definite integral of the spline from a to b.
    ///
    /// Note: This method uses scalar computation as integration bounds
    /// are typically scalar values, not batched tensors.
    pub fn integrate(&self, a: f64, b: f64) -> InterpolateResult<f64> {
        if a < self.x_min || b > self.x_max {
            return Err(InterpolateError::OutOfDomain {
                point: if a < self.x_min { a } else { b },
                min: self.x_min,
                max: self.x_max,
                context: "CubicSpline::integrate".to_string(),
            });
        }

        if a >= b {
            return Ok(0.0);
        }

        // Get x and coefficient data for integration
        let x_data: Vec<f64> = self.x.contiguous().to_vec();
        let a_data: Vec<f64> = self.a.contiguous().to_vec();
        let b_data: Vec<f64> = self.b.contiguous().to_vec();
        let c_data: Vec<f64> = self.c.contiguous().to_vec();
        let d_data: Vec<f64> = self.d.contiguous().to_vec();

        let mut result = 0.0;
        let mut current = a;

        while current < b {
            let idx = self.find_interval(&x_data, current);
            let x_end = x_data[idx + 1].min(b);

            // Integrate S_i(x) from current to x_end
            let dx0 = current - x_data[idx];
            let dx1 = x_end - x_data[idx];

            let integral_at = |dx: f64| -> f64 {
                a_data[idx] * dx
                    + b_data[idx] * dx * dx / 2.0
                    + c_data[idx] * dx * dx * dx / 3.0
                    + d_data[idx] * dx * dx * dx * dx / 4.0
            };

            result += integral_at(dx1) - integral_at(dx0);
            current = x_end;
        }

        Ok(result)
    }

    /// Find the interval index for a given x value (binary search).
    fn find_interval(&self, x_data: &[f64], xi: f64) -> usize {
        let mut lo = 0;
        let mut hi = self.n - 1;

        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if x_data[mid] <= xi {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::cubic_spline::SplineBoundary;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_evaluate_at_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);
        assert!((y_result[1] - 1.0).abs() < 1e-10);
        assert!((y_result[2] - 0.0).abs() < 1e-10);
        assert!((y_result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_linear() {
        let (device, client) = setup();

        // y = 2x + 1, derivative should be 2
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let dy = spline.derivative(&client, &x_test).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        for val in dy_result {
            assert!(
                (val - 2.0).abs() < 0.1,
                "Derivative should be ~2, got {}",
                val
            );
        }
    }

    #[test]
    fn test_second_derivative_quadratic() {
        let (device, client) = setup();

        // y = x^2, second derivative should be 2
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[1.5, 2.0, 2.5], &[3], &device);
        let d2y = spline.second_derivative(&client, &x_test).unwrap();
        let d2y_result: Vec<f64> = d2y.to_vec();

        for val in d2y_result {
            assert!(
                (val - 2.0).abs() < 0.5,
                "Second derivative should be ~2, got {}",
                val
            );
        }
    }

    #[test]
    fn test_integrate_linear() {
        let (device, client) = setup();

        // y = 2x, integral from 0 to 2 is 4
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(0.0, 2.0).unwrap();
        assert!(
            (integral - 4.0).abs() < 1e-8,
            "Integral should be 4, got {}",
            integral
        );
    }

    #[test]
    fn test_integrate_constant() {
        let (device, client) = setup();

        // y = 3, integral from 0 to 4 is 12
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[3.0, 3.0, 3.0, 3.0, 3.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(0.0, 4.0).unwrap();
        assert!(
            (integral - 12.0).abs() < 1e-8,
            "Integral should be 12, got {}",
            integral
        );
    }

    #[test]
    fn test_out_of_bounds_clamps() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Out-of-bounds queries are clamped to boundary intervals
        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5, 2.5], &[2], &device);
        let result = spline.evaluate(&client, &x_oob).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Results should be computed using the boundary intervals
        // (not an error, algorithm clamps indices)
        assert_eq!(result_data.len(), 2);
    }
}
