//! 1D interpolation methods.
//!
//! Provides linear, nearest neighbor, and cubic interpolation for 1D data.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Interpolation method for 1D data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation between adjacent points.
    Linear,
    /// Cubic interpolation using 4 neighboring points.
    Cubic,
}

/// 1D interpolator that stores precomputed data for fast evaluation.
///
/// # Example
///
/// ```ignore
/// use solvr::interpolate::{Interp1d, InterpMethod};
///
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
/// let y = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);
///
/// let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear)?;
/// let y_new = interp.evaluate(&client, &x_new)?;
/// ```
pub struct Interp1d<R: Runtime> {
    /// Sorted x coordinates (knots).
    x: Tensor<R>,
    /// Corresponding y values.
    y: Tensor<R>,
    /// Interpolation method.
    method: InterpMethod,
    /// Number of data points.
    n: usize,
    /// Minimum x value (for bounds checking).
    x_min: f64,
    /// Maximum x value (for bounds checking).
    x_max: f64,
}

impl<R: Runtime> Interp1d<R> {
    /// Create a new 1D interpolator.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    /// * `method` - Interpolation method to use
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - x and y have different lengths
    /// - x has fewer than 2 points (or 4 for cubic)
    /// - x values are not strictly increasing
    pub fn new<C: RuntimeClient<R>>(
        _client: &C,
        x: &Tensor<R>,
        y: &Tensor<R>,
        method: InterpMethod,
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
                context: "Interp1d::new".to_string(),
            });
        }

        // Check minimum points required
        let min_points = match method {
            InterpMethod::Nearest | InterpMethod::Linear => 2,
            InterpMethod::Cubic => 4,
        };

        if n < min_points {
            return Err(InterpolateError::InsufficientData {
                required: min_points,
                actual: n,
                context: format!("{:?} interpolation", method),
            });
        }

        // Get x values to check monotonicity and bounds
        let x_data: Vec<f64> = x.to_vec();

        // Check strictly increasing
        for i in 1..n {
            if x_data[i] <= x_data[i - 1] {
                return Err(InterpolateError::NotMonotonic {
                    context: "Interp1d::new".to_string(),
                });
            }
        }

        let x_min = x_data[0];
        let x_max = x_data[n - 1];

        Ok(Self {
            x: x.clone(),
            y: y.clone(),
            method,
            n,
            x_min,
            x_max,
        })
    }

    /// Evaluate the interpolator at new x coordinates.
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

        // Get data as vectors for CPU computation
        let x_data: Vec<f64> = self.x.to_vec();
        let y_data: Vec<f64> = self.y.to_vec();
        let x_new_data: Vec<f64> = x_new.to_vec();

        // Compute interpolated values
        let mut y_new_data = Vec::with_capacity(x_new_data.len());

        for &xi in &x_new_data {
            // Bounds check
            if xi < self.x_min || xi > self.x_max {
                return Err(InterpolateError::OutOfDomain {
                    point: xi,
                    min: self.x_min,
                    max: self.x_max,
                    context: "Interp1d::evaluate".to_string(),
                });
            }

            // Find interval index using binary search
            let idx = self.find_interval(&x_data, xi);

            // Interpolate based on method
            let yi = match self.method {
                InterpMethod::Nearest => self.nearest(&x_data, &y_data, idx, xi),
                InterpMethod::Linear => self.linear(&x_data, &y_data, idx, xi),
                InterpMethod::Cubic => self.cubic(&x_data, &y_data, idx, xi),
            };

            y_new_data.push(yi);
        }

        // Convert result back to tensor
        let device = x_new.device();
        Ok(Tensor::from_slice(&y_new_data, &[y_new_data.len()], device))
    }

    /// Find the interval index for a given x value.
    /// Returns i such that x[i] <= xi < x[i+1] (or i = n-2 if xi == x[n-1]).
    fn find_interval(&self, x_data: &[f64], xi: f64) -> usize {
        // Binary search for the interval
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

    /// Nearest neighbor interpolation.
    fn nearest(&self, x_data: &[f64], y_data: &[f64], idx: usize, xi: f64) -> f64 {
        let x0 = x_data[idx];
        let x1 = x_data[idx + 1];

        // Return y value of nearest point
        if (xi - x0) <= (x1 - xi) {
            y_data[idx]
        } else {
            y_data[idx + 1]
        }
    }

    /// Linear interpolation.
    fn linear(&self, x_data: &[f64], y_data: &[f64], idx: usize, xi: f64) -> f64 {
        let x0 = x_data[idx];
        let x1 = x_data[idx + 1];
        let y0 = y_data[idx];
        let y1 = y_data[idx + 1];

        // Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        let t = (xi - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }

    /// Cubic interpolation using Catmull-Rom spline.
    fn cubic(&self, x_data: &[f64], y_data: &[f64], idx: usize, xi: f64) -> f64 {
        // Get 4 points for cubic interpolation
        // Clamp indices at boundaries
        let i0 = if idx == 0 { 0 } else { idx - 1 };
        let i1 = idx;
        let i2 = idx + 1;
        let i3 = if idx + 2 >= self.n { self.n - 1 } else { idx + 2 };

        let x0 = x_data[i0];
        let x1 = x_data[i1];
        let x2 = x_data[i2];
        let x3 = x_data[i3];

        let y0 = y_data[i0];
        let y1 = y_data[i1];
        let y2 = y_data[i2];
        let y3 = y_data[i3];

        // Parameter t in [0, 1] for the interval [x1, x2]
        let t = (xi - x1) / (x2 - x1);
        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom basis functions
        // Handle non-uniform spacing by using tangent approximations
        let h1 = x2 - x1;
        let h0 = x1 - x0;
        let h2 = x3 - x2;

        // Tangents at x1 and x2 (finite difference approximation)
        let m1 = if i0 == i1 {
            (y2 - y1) / h1
        } else {
            0.5 * ((y2 - y1) / h1 + (y1 - y0) / h0)
        };

        let m2 = if i2 == i3 {
            (y2 - y1) / h1
        } else {
            0.5 * ((y3 - y2) / h2 + (y2 - y1) / h1)
        };

        // Hermite basis functions
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        // Cubic Hermite interpolation
        h00 * y1 + h10 * h1 * m1 + h01 * y2 + h11 * h1 * m2
    }

    /// Returns the interpolation method.
    pub fn method(&self) -> InterpMethod {
        self.method
    }

    /// Returns the number of data points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the interpolator has no data points.
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

    #[test]
    fn test_linear_interpolation_basic() {
        let (device, client) = setup();

        // y = 2x line
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        // Test at midpoints
        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 1.0).abs() < 1e-10); // 0.5 -> 1.0
        assert!((y_result[1] - 3.0).abs() < 1e-10); // 1.5 -> 3.0
        assert!((y_result[2] - 5.0).abs() < 1e-10); // 2.5 -> 5.0
    }

    #[test]
    fn test_linear_interpolation_at_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0, 30.0], &[3], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        // Test at exact knot positions
        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 10.0).abs() < 1e-10);
        assert!((y_result[1] - 20.0).abs() < 1e-10);
        assert!((y_result[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 10.0, 20.0, 30.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Nearest).unwrap();

        // Test nearest neighbor behavior
        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.3, 0.5, 0.7, 1.9], &[4], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);  // 0.3 -> nearer to 0 -> 0.0
        assert!((y_result[1] - 0.0).abs() < 1e-10);  // 0.5 -> ties go to left -> 0.0
        assert!((y_result[2] - 10.0).abs() < 1e-10); // 0.7 -> nearer to 1 -> 10.0
        assert!((y_result[3] - 20.0).abs() < 1e-10); // 1.9 -> nearer to 2 -> 20.0
    }

    #[test]
    fn test_cubic_interpolation() {
        let (device, client) = setup();

        // y = x^2 (quadratic should be exact with cubic interp at midpoints)
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Cubic).unwrap();

        // Test at 1.5 (should be close to 2.25)
        let x_new = Tensor::<CpuRuntime>::from_slice(&[1.5], &[1], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        // Cubic should be reasonably close to quadratic
        assert!((y_result[0] - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        // Test below range
        let x_new = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1], &device);
        let result = interp.evaluate(&client, &x_new);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));

        // Test above range
        let x_new = Tensor::<CpuRuntime>::from_slice(&[2.5], &[1], &device);
        let result = interp.evaluate(&client, &x_new);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    #[test]
    fn test_non_monotonic_error() {
        let (device, client) = setup();

        // x is not strictly increasing
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.5, 2.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);

        let result = Interp1d::new(&client, &x, &y, InterpMethod::Linear);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = Interp1d::new(&client, &x, &y, InterpMethod::Linear);
        assert!(matches!(result, Err(InterpolateError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_insufficient_data_error() {
        let (device, client) = setup();

        // Only 1 point - need at least 2 for linear
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = Interp1d::new(&client, &x, &y, InterpMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));

        // Only 3 points - need at least 4 for cubic
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = Interp1d::new(&client, &x, &y, InterpMethod::Cubic);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_bounds_method() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 5.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.0], &[3], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();
        let (x_min, x_max) = interp.bounds();

        assert!((x_min - 1.0).abs() < 1e-10);
        assert!((x_max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_uniform_spacing() {
        let (device, client) = setup();

        // Non-uniform spacing
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.1, 0.5, 2.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.2, 1.0, 4.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        // Test in different intervals
        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.05, 0.3, 1.25], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        // 0.05 is halfway between 0 and 0.1, so (0.0 + 0.2) / 2 = 0.1
        assert!((y_result[0] - 0.1).abs() < 1e-10);

        // 0.3 is (0.3-0.1)/(0.5-0.1) = 0.5 of the way from 0.1 to 0.5
        // So y = 0.2 + 0.5 * (1.0 - 0.2) = 0.2 + 0.4 = 0.6
        assert!((y_result[1] - 0.6).abs() < 1e-10);
    }
}
