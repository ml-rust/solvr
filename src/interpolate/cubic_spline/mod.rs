//! Cubic spline interpolation.
//!
//! Provides cubic spline interpolation with various boundary conditions:
//! - Natural: zero second derivative at endpoints
//! - Clamped: specified first derivative at endpoints
//! - Not-a-knot: third derivative continuous at second and second-to-last points

mod coefficients;
mod evaluate;

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
    /// X coordinates (knots).
    pub(crate) x: Tensor<R>,
    /// Polynomial coefficients a_i (= y_i) as tensor.
    pub(crate) a: Tensor<R>,
    /// Polynomial coefficients b_i as tensor.
    pub(crate) b: Tensor<R>,
    /// Polynomial coefficients c_i as tensor (length n).
    pub(crate) c: Tensor<R>,
    /// Polynomial coefficients d_i as tensor.
    pub(crate) d: Tensor<R>,
    /// Number of data points.
    pub(crate) n: usize,
    /// Minimum x value.
    pub(crate) x_min: f64,
    /// Maximum x value.
    pub(crate) x_max: f64,
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
        client: &C,
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

        // Get data as vectors for coefficient computation (construction time)
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let y_data: Vec<f64> = y.contiguous().to_vec();

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

        // Compute spline coefficients (tridiagonal solver - inherently sequential)
        let (a_vec, b_vec, c_vec, d_vec) =
            coefficients::compute_coefficients(&x_data, &y_data, &boundary)?;

        // Store coefficients as tensors for GPU evaluation
        let device = client.device();
        let a = Tensor::from_slice(&a_vec, &[n], device);
        let b = Tensor::from_slice(&b_vec, &[n - 1], device);
        let c = Tensor::from_slice(&c_vec, &[n], device);
        let d = Tensor::from_slice(&d_vec, &[n - 1], device);

        Ok(Self {
            x: x.clone(),
            a,
            b,
            c,
            d,
            n,
            x_min,
            x_max,
        })
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

    #[test]
    fn test_natural_spline_at_knots() {
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
    fn test_natural_spline_second_derivative_endpoints() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 1.5, 3.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        let endpoints = Tensor::<CpuRuntime>::from_slice(&[0.0, 3.0], &[2], &device);
        let d2y = spline.second_derivative(&client, &endpoints).unwrap();
        let d2y_result: Vec<f64> = d2y.to_vec();

        assert!(d2y_result[0].abs() < 1e-10);
        assert!(d2y_result[1].abs() < 1e-10);
    }

    #[test]
    fn test_clamped_spline_derivative() {
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

        let endpoints = Tensor::<CpuRuntime>::from_slice(&[0.0, 3.0], &[2], &device);
        let dy = spline.derivative(&client, &endpoints).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        assert!((dy_result[0] - left_deriv).abs() < 1e-8);
        assert!((dy_result[1] - right_deriv).abs() < 1e-8);
    }

    #[test]
    fn test_not_a_knot_spline() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 1.5, 3.0, 2.5], &[5], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::NotAKnot).unwrap();

        let y_test = spline.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 1.0).abs() < 1e-10);
        assert!((y_result[1] - 2.0).abs() < 1e-10);
        assert!((y_result[2] - 1.5).abs() < 1e-10);
        assert!((y_result[3] - 3.0).abs() < 1e-10);
        assert!((y_result[4] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_len_and_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 5.0, 10.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let spline = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural).unwrap();

        assert_eq!(spline.len(), 4);
        assert!(!spline.is_empty());

        let (min, max) = spline.bounds();
        assert!((min - 1.0).abs() < 1e-14);
        assert!((max - 10.0).abs() < 1e-14);
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
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_insufficient_data_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = CubicSpline::new(&client, &x, &y, SplineBoundary::Natural);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));
    }
}
