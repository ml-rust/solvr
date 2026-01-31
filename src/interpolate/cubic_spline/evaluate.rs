//! Cubic spline evaluation methods.
//!
//! Provides methods for evaluating the spline, its derivatives, and integrals.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::CubicSpline;

impl<R: Runtime> CubicSpline<R> {
    /// Evaluate the spline at new x coordinates.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x_new` - 1D tensor of x coordinates to evaluate at
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated y values.
    pub fn evaluate<C: RuntimeClient<R>>(
        &self,
        _client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let x_new_shape = x_new.shape();
        if x_new_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x_new".to_string(),
                message: "x_new must be a 1D tensor".to_string(),
            });
        }

        let x_new_data: Vec<f64> = x_new.to_vec();

        let mut y_new_data = Vec::with_capacity(x_new_data.len());

        for &xi in &x_new_data {
            // Bounds check
            if xi < self.x_min || xi > self.x_max {
                return Err(InterpolateError::OutOfDomain {
                    point: xi,
                    min: self.x_min,
                    max: self.x_max,
                    context: "CubicSpline::evaluate".to_string(),
                });
            }

            // Find interval
            let idx = self.find_interval(xi);

            // Evaluate polynomial
            let dx = xi - self.x_data[idx];
            let yi = self.a[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]));

            y_new_data.push(yi);
        }

        let device = x_new.device();
        Ok(Tensor::from_slice(&y_new_data, &[y_new_data.len()], device))
    }

    /// Evaluate the first derivative of the spline.
    pub fn derivative<C: RuntimeClient<R>>(
        &self,
        _client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let x_new_data: Vec<f64> = x_new.to_vec();

        let mut dy_data = Vec::with_capacity(x_new_data.len());

        for &xi in &x_new_data {
            if xi < self.x_min || xi > self.x_max {
                return Err(InterpolateError::OutOfDomain {
                    point: xi,
                    min: self.x_min,
                    max: self.x_max,
                    context: "CubicSpline::derivative".to_string(),
                });
            }

            let idx = self.find_interval(xi);
            let dx = xi - self.x_data[idx];

            // S'(x) = b + 2*c*dx + 3*d*dx^2
            let dyi = self.b[idx] + dx * (2.0 * self.c[idx] + 3.0 * self.d[idx] * dx);
            dy_data.push(dyi);
        }

        let device = x_new.device();
        Ok(Tensor::from_slice(&dy_data, &[dy_data.len()], device))
    }

    /// Evaluate the second derivative of the spline.
    pub fn second_derivative<C: RuntimeClient<R>>(
        &self,
        _client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let x_new_data: Vec<f64> = x_new.to_vec();

        let mut d2y_data = Vec::with_capacity(x_new_data.len());

        for &xi in &x_new_data {
            if xi < self.x_min || xi > self.x_max {
                return Err(InterpolateError::OutOfDomain {
                    point: xi,
                    min: self.x_min,
                    max: self.x_max,
                    context: "CubicSpline::second_derivative".to_string(),
                });
            }

            let idx = self.find_interval(xi);
            let dx = xi - self.x_data[idx];

            // S''(x) = 2*c + 6*d*dx
            let d2yi = 2.0 * self.c[idx] + 6.0 * self.d[idx] * dx;
            d2y_data.push(d2yi);
        }

        let device = x_new.device();
        Ok(Tensor::from_slice(&d2y_data, &[d2y_data.len()], device))
    }

    /// Compute the definite integral of the spline from a to b.
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

        let mut result = 0.0;
        let mut current = a;

        while current < b {
            let idx = self.find_interval(current);
            let x_end = self.x_data[idx + 1].min(b);

            // Integrate S_i(x) from current to x_end
            // ∫ (a + b*dx + c*dx^2 + d*dx^3) dx
            // = a*dx + b*dx^2/2 + c*dx^3/3 + d*dx^4/4

            let dx0 = current - self.x_data[idx];
            let dx1 = x_end - self.x_data[idx];

            let integral_at = |dx: f64| -> f64 {
                self.a[idx] * dx
                    + self.b[idx] * dx * dx / 2.0
                    + self.c[idx] * dx * dx * dx / 3.0
                    + self.d[idx] * dx * dx * dx * dx / 4.0
            };

            result += integral_at(dx1) - integral_at(dx0);
            current = x_end;
        }

        Ok(result)
    }

    /// Find the interval index for a given x value.
    pub(super) fn find_interval(&self, xi: f64) -> usize {
        let mut lo = 0;
        let mut hi = self.n - 1;

        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.x_data[mid] <= xi {
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
    fn test_out_of_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1], &device);
        let result = spline.evaluate(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }
}
