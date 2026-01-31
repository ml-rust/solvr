//! 1D interpolation methods.
//!
//! Provides linear, nearest neighbor, and cubic interpolation for 1D data.

mod methods;

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
    pub(crate) x: Tensor<R>,
    /// Corresponding y values.
    pub(crate) y: Tensor<R>,
    /// Interpolation method.
    pub(crate) method: InterpMethod,
    /// Number of data points.
    pub(crate) n: usize,
    /// Minimum x value (for bounds checking).
    pub(crate) x_min: f64,
    /// Maximum x value (for bounds checking).
    pub(crate) x_max: f64,
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
    fn test_at_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0, 30.0], &[3], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 10.0).abs() < 1e-10);
        assert!((y_result[1] - 20.0).abs() < 1e-10);
        assert!((y_result[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_monotonic_error() {
        let (device, client) = setup();

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
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_insufficient_data_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = Interp1d::new(&client, &x, &y, InterpMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));

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

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.1, 0.5, 2.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.2, 1.0, 4.0], &[4], &device);

        let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear).unwrap();

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.05, 0.3], &[2], &device);
        let y_new = interp.evaluate(&client, &x_new).unwrap();
        let y_result: Vec<f64> = y_new.to_vec();

        assert!((y_result[0] - 0.1).abs() < 1e-10);
        assert!((y_result[1] - 0.6).abs() < 1e-10);
    }
}
