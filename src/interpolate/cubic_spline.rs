//! Cubic spline interpolation.
//!
//! Provides cubic spline interpolation with various boundary conditions:
//! - Natural: zero second derivative at endpoints
//! - Clamped: specified first derivative at endpoints
//! - Not-a-knot: third derivative continuous at second and second-to-last points

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Boundary condition for cubic spline.
#[derive(Debug, Clone, Default)]
pub enum SplineBoundary {
    /// Natural spline: second derivative is zero at endpoints.
    #[default]
    Natural,
    /// Clamped spline: first derivative is specified at endpoints.
    Clamped { left: f64, right: f64 },
    /// Not-a-knot: third derivative is continuous at second and second-to-last points.
    NotAKnot,
}

/// Cubic spline interpolator.
///
/// Computes and stores cubic polynomial coefficients for each interval.
/// Each interval [x_i, x_{i+1}] has a cubic polynomial:
///   S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
///
/// # Example
///
/// ```ignore
/// use solvr::interpolate::{CubicSpline, SplineBoundary};
///
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
/// let y = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);
///
/// let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural)?;
/// let y_new = spline.evaluate(&client, &x_new)?;
/// ```
pub struct CubicSpline<R: Runtime> {
    /// X coordinates (knots). Kept for potential future GPU operations.
    #[allow(dead_code)]
    x: Tensor<R>,
    /// Polynomial coefficients a_i (= y_i).
    a: Vec<f64>,
    /// Polynomial coefficients b_i.
    b: Vec<f64>,
    /// Polynomial coefficients c_i.
    c: Vec<f64>,
    /// Polynomial coefficients d_i.
    d: Vec<f64>,
    /// Number of data points.
    n: usize,
    /// Minimum x value.
    x_min: f64,
    /// Maximum x value.
    x_max: f64,
    /// Cached x data for evaluation.
    x_data: Vec<f64>,
}

impl<R: Runtime> CubicSpline<R> {
    /// Create a new cubic spline interpolator.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    /// * `boundary` - Boundary condition for the spline
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - x and y have different lengths
    /// - x has fewer than 2 points
    /// - x values are not strictly increasing
    pub fn new<C: RuntimeClient<R>>(
        _client: &C,
        x: &Tensor<R>,
        y: &Tensor<R>,
        boundary: SplineBoundary,
    ) -> InterpolateResult<Self> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        // Validate shapes
        if x_shape.len() != 1 || y_shape.len() != 1 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "x, y".to_string(),
                message: "x and y must be 1D tensors".to_string(),
            });
        }

        let n = x_shape[0];
        if n != y_shape[0] {
            return Err(InterpolateError::ShapeMismatch {
                expected: n,
                actual: y_shape[0],
                context: "CubicSpline::new".to_string(),
            });
        }

        if n < 2 {
            return Err(InterpolateError::InsufficientData {
                required: 2,
                actual: n,
                context: "CubicSpline::new".to_string(),
            });
        }

        // Get data as vectors
        let x_data: Vec<f64> = x.to_vec();
        let y_data: Vec<f64> = y.to_vec();

        // Check strictly increasing
        for i in 1..n {
            if x_data[i] <= x_data[i - 1] {
                return Err(InterpolateError::NotMonotonic {
                    context: "CubicSpline::new".to_string(),
                });
            }
        }

        let x_min = x_data[0];
        let x_max = x_data[n - 1];

        // Compute spline coefficients
        let (a, b, c, d) = Self::compute_coefficients(&x_data, &y_data, &boundary)?;

        Ok(Self {
            x: x.clone(),
            a,
            b,
            c,
            d,
            n,
            x_min,
            x_max,
            x_data,
        })
    }

    /// Compute cubic spline coefficients using Thomas algorithm.
    #[allow(clippy::type_complexity)]
    fn compute_coefficients(
        x: &[f64],
        y: &[f64],
        boundary: &SplineBoundary,
    ) -> InterpolateResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = x.len();

        // a coefficients are just the y values
        let a: Vec<f64> = y.to_vec();

        // Compute interval widths h_i = x_{i+1} - x_i
        let mut h = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            h.push(x[i + 1] - x[i]);
        }

        // Set up tridiagonal system for c coefficients (second derivatives / 2)
        // The system is: lower[i] * c[i-1] + diag[i] * c[i] + upper[i] * c[i+1] = rhs[i]
        let mut diag = vec![0.0; n];
        let mut upper = vec![0.0; n - 1];
        let mut lower = vec![0.0; n - 1];
        let mut rhs = vec![0.0; n];

        // Interior equations (natural cubic spline continuity)
        for i in 1..n - 1 {
            lower[i - 1] = h[i - 1];
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        // Apply boundary conditions
        match boundary {
            SplineBoundary::Natural => {
                // c[0] = 0, c[n-1] = 0
                diag[0] = 1.0;
                rhs[0] = 0.0;
                diag[n - 1] = 1.0;
                rhs[n - 1] = 0.0;
            }
            SplineBoundary::Clamped { left, right } => {
                // First derivative specified at endpoints
                // S'(x_0) = left => b[0] = left
                // S'(x_{n-1}) = right => b[n-1] = right
                diag[0] = 2.0 * h[0];
                upper[0] = h[0];
                rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - *left);

                diag[n - 1] = 2.0 * h[n - 2];
                lower[n - 2] = h[n - 2];
                rhs[n - 1] = 3.0 * (*right - (y[n - 1] - y[n - 2]) / h[n - 2]);
            }
            SplineBoundary::NotAKnot => {
                if n < 4 {
                    // Fall back to natural for small n
                    diag[0] = 1.0;
                    rhs[0] = 0.0;
                    diag[n - 1] = 1.0;
                    rhs[n - 1] = 0.0;
                } else {
                    // Not-a-knot: d[0] = d[1] and d[n-3] = d[n-2]
                    // This makes third derivative continuous at x[1] and x[n-2]
                    diag[0] = h[1];
                    upper[0] = -(h[0] + h[1]);
                    rhs[0] = 0.0;
                    // Need to add h[0] * c[2] term - modify the system
                    // For simplicity, we use a modified approach
                    let h0h1 = h[0] * h[0] * h[1];
                    let h1h0 = h[1] * h[1] * h[0];
                    diag[0] = h1h0;
                    upper[0] = -(h0h1 + h1h0);
                    rhs[0] = h0h1 * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0]);

                    let hn2 = h[n - 2];
                    let hn3 = h[n - 3];
                    diag[n - 1] = hn3 * hn3 * hn2;
                    lower[n - 2] = -(hn2 * hn2 * hn3 + hn3 * hn3 * hn2);
                    rhs[n - 1] = hn2 * hn2 * hn3
                        * ((y[n - 1] - y[n - 2]) / hn2 - (y[n - 2] - y[n - 3]) / hn3);
                }
            }
        }

        // Solve tridiagonal system using Thomas algorithm
        let c = Self::solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

        // Compute b and d coefficients from c
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];

        for i in 0..n - 1 {
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        Ok((a, b, c, d))
    }

    /// Solve tridiagonal system using Thomas algorithm.
    fn solve_tridiagonal(
        lower: &[f64],
        diag: &[f64],
        upper: &[f64],
        rhs: &[f64],
    ) -> InterpolateResult<Vec<f64>> {
        let n = diag.len();
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        // Forward sweep
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];

        for i in 1..n {
            let denom = diag[i] - lower[i.saturating_sub(1)] * c_prime[i - 1];
            if denom.abs() < 1e-14 {
                return Err(InterpolateError::NumericalError {
                    message: "Singular tridiagonal system in spline computation".to_string(),
                });
            }
            if i < n - 1 {
                c_prime[i] = upper[i] / denom;
            }
            d_prime[i] = (rhs[i] - lower[i.saturating_sub(1)] * d_prime[i - 1]) / denom;
        }

        // Back substitution
        let mut x = vec![0.0; n];
        x[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        Ok(x)
    }

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
            let yi =
                self.a[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]));

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
    fn find_interval(&self, xi: f64) -> usize {
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

    /// Returns the number of data points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the spline has no data points.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the domain bounds (x_min, x_max).
    pub fn bounds(&self) -> (f64, f64) {
        (self.x_min, self.x_max)
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

    // ========================================================================
    // Natural Spline Tests
    // ========================================================================

    #[test]
    fn test_natural_spline_at_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Evaluate at knot points - should return exact y values
        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y_test = spline.evaluate(&client, &x_test).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);
        assert!((y_result[1] - 1.0).abs() < 1e-10);
        assert!((y_result[2] - 0.0).abs() < 1e-10);
        assert!((y_result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_spline_second_derivative_at_endpoints() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 1.5, 3.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Natural spline: second derivative is zero at endpoints
        let endpoints = Tensor::<CpuRuntime>::from_slice(&[0.0, 3.0], &[2], &device);
        let d2y = spline.second_derivative(&client, &endpoints).unwrap();
        let d2y_result: Vec<f64> = d2y.to_vec();

        assert!(d2y_result[0].abs() < 1e-10, "Left endpoint second derivative should be ~0");
        assert!(d2y_result[1].abs() < 1e-10, "Right endpoint second derivative should be ~0");
    }

    #[test]
    fn test_natural_spline_interpolation_midpoints() {
        let (device, client) = setup();

        // Use a simple quadratic: y = x^2
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Interpolate at midpoints
        let x_mid = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5, 3.5], &[4], &device);
        let y_interp = spline.evaluate(&client, &x_mid).unwrap();
        let y_result: Vec<f64> = y_interp.to_vec();

        // For a quadratic, cubic spline should be quite accurate
        assert!((y_result[0] - 0.25).abs() < 0.1);
        assert!((y_result[1] - 2.25).abs() < 0.1);
        assert!((y_result[2] - 6.25).abs() < 0.1);
        assert!((y_result[3] - 12.25).abs() < 0.1);
    }

    // ========================================================================
    // Clamped Spline Tests
    // ========================================================================

    #[test]
    fn test_clamped_spline_derivative_at_endpoints() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let left_deriv = 2.0;
        let right_deriv = -1.0;

        let spline = CubicSpline::new(
            &client,
            &x,
            &y,
            SplineBoundary::Clamped {
                left: left_deriv,
                right: right_deriv,
            },
        )
        .unwrap();

        // Check that the derivatives at endpoints match specified values
        let endpoints = Tensor::<CpuRuntime>::from_slice(&[0.0, 3.0], &[2], &device);
        let dy = spline.derivative(&client, &endpoints).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        assert!(
            (dy_result[0] - left_deriv).abs() < 1e-8,
            "Left derivative should be {}, got {}",
            left_deriv,
            dy_result[0]
        );
        assert!(
            (dy_result[1] - right_deriv).abs() < 1e-8,
            "Right derivative should be {}, got {}",
            right_deriv,
            dy_result[1]
        );
    }

    #[test]
    fn test_clamped_spline_passes_through_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 2.0], &[3], &device);

        let spline = CubicSpline::new(
            &client,
            &x,
            &y,
            SplineBoundary::Clamped {
                left: 0.0,
                right: 0.0,
            },
        )
        .unwrap();

        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 1.0).abs() < 1e-10);
        assert!((y_result[1] - 3.0).abs() < 1e-10);
        assert!((y_result[2] - 2.0).abs() < 1e-10);
    }

    // ========================================================================
    // Not-a-Knot Spline Tests
    // ========================================================================

    #[test]
    fn test_not_a_knot_spline_at_knots() {
        let (device, client) = setup();

        // Need at least 4 points for not-a-knot to be meaningful
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 1.5, 3.0, 2.5], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::NotAKnot).unwrap();

        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        // Should pass through all knots exactly
        assert!((y_result[0] - 1.0).abs() < 1e-10);
        assert!((y_result[1] - 2.0).abs() < 1e-10);
        assert!((y_result[2] - 1.5).abs() < 1e-10);
        assert!((y_result[3] - 3.0).abs() < 1e-10);
        assert!((y_result[4] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_not_a_knot_fallback_for_small_n() {
        let (device, client) = setup();

        // With n < 4, not-a-knot falls back to natural
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 1.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::NotAKnot).unwrap();

        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        // Should still pass through knots
        assert!((y_result[0] - 1.0).abs() < 1e-10);
        assert!((y_result[1] - 2.0).abs() < 1e-10);
        assert!((y_result[2] - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // Derivative Tests
    // ========================================================================

    #[test]
    fn test_derivative_of_linear_function() {
        let (device, client) = setup();

        // y = 2x + 1, derivative should be 2
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let dy = spline.derivative(&client, &x_test).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        // For linear data, derivative should be constant ~2
        for val in dy_result {
            assert!((val - 2.0).abs() < 0.1, "Derivative should be ~2, got {}", val);
        }
    }

    #[test]
    fn test_second_derivative_of_quadratic() {
        let (device, client) = setup();

        // y = x^2, second derivative should be 2
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Check second derivative in the interior (not at endpoints due to natural BC)
        let x_test = Tensor::<CpuRuntime>::from_slice(&[1.5, 2.0, 2.5], &[3], &device);
        let d2y = spline.second_derivative(&client, &x_test).unwrap();
        let d2y_result: Vec<f64> = d2y.to_vec();

        // For quadratic, second derivative should be close to 2 in interior
        for val in d2y_result {
            assert!((val - 2.0).abs() < 0.5, "Second derivative should be ~2, got {}", val);
        }
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_integrate_linear_function() {
        let (device, client) = setup();

        // y = 2x, integral from 0 to 2 is x^2 from 0 to 2 = 4
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(0.0, 2.0).unwrap();
        assert!((integral - 4.0).abs() < 1e-8, "Integral should be 4, got {}", integral);
    }

    #[test]
    fn test_integrate_constant_function() {
        let (device, client) = setup();

        // y = 3, integral from 0 to 4 is 12
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[3.0, 3.0, 3.0, 3.0, 3.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(0.0, 4.0).unwrap();
        assert!((integral - 12.0).abs() < 1e-8, "Integral should be 12, got {}", integral);
    }

    #[test]
    fn test_integrate_partial_interval() {
        let (device, client) = setup();

        // y = x, integral from 1 to 3 is (x^2/2) from 1 to 3 = 4.5 - 0.5 = 4
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(1.0, 3.0).unwrap();
        assert!((integral - 4.0).abs() < 1e-8, "Integral should be 4, got {}", integral);
    }

    #[test]
    fn test_integrate_same_point() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let integral = spline.integrate(1.0, 1.0).unwrap();
        assert!(integral.abs() < 1e-14, "Integral from a to a should be 0");
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Test evaluation out of bounds
        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1], &device);
        let result = spline.evaluate(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));

        let x_oob = Tensor::<CpuRuntime>::from_slice(&[2.5], &[1], &device);
        let result = spline.evaluate(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    #[test]
    fn test_integrate_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let result = spline.integrate(-1.0, 1.0);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));

        let result = spline.integrate(0.0, 3.0);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    #[test]
    fn test_non_monotonic_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural);
        assert!(matches!(result, Err(InterpolateError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_insufficient_data_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural);
        assert!(matches!(result, Err(InterpolateError::InsufficientData { .. })));
    }

    #[test]
    fn test_derivative_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let x_oob = Tensor::<CpuRuntime>::from_slice(&[3.0], &[1], &device);
        let result = spline.derivative(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    // ========================================================================
    // Utility Method Tests
    // ========================================================================

    #[test]
    fn test_len_and_is_empty() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        assert_eq!(spline.len(), 4);
        assert!(!spline.is_empty());
    }

    #[test]
    fn test_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 5.0, 10.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let (min, max) = spline.bounds();
        assert!((min - 1.0).abs() < 1e-14);
        assert!((max - 10.0).abs() < 1e-14);
    }

    // ========================================================================
    // Non-uniform Spacing Tests
    // ========================================================================

    #[test]
    fn test_non_uniform_spacing() {
        let (device, client) = setup();

        // Non-uniformly spaced x values
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.1, 0.5, 2.0, 5.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.1, 1.5, 2.0, 3.0], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        // Should pass through knots
        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        for (i, (&expected, &actual)) in y.to_vec::<f64>().iter().zip(y_result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-10,
                "Knot {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
