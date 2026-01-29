//! Akima spline interpolation.
//!
//! Akima spline is a piecewise cubic interpolation method that:
//! - Is less sensitive to outliers than traditional cubic splines
//! - Has continuous first derivative (C1)
//! - Uses locally-weighted slopes to reduce oscillation
//!
//! # When to Use Akima vs Other Methods
//!
//! | Property          | Akima                | CubicSpline          | PCHIP               |
//! |-------------------|----------------------|----------------------|---------------------|
//! | Smoothness        | C1 (continuous 1st)  | C2 (continuous 2nd)  | C1 (continuous 1st) |
//! | Outlier handling  | Robust               | Sensitive            | Moderate            |
//! | Oscillation       | Minimal              | Can oscillate        | None                |
//! | Monotonicity      | Not guaranteed       | Not guaranteed       | Preserved           |
//!
//! # Algorithm
//!
//! The Akima method computes slopes at each point using a weighted average:
//! 1. Compute slopes between all adjacent points
//! 2. At each interior point, compute slope as weighted average based on
//!    differences between adjacent slopes (weights reduce influence of outliers)
//! 3. Use Hermite basis functions to construct cubic polynomial

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::hermite_core::{
    HermiteData, derivative_hermite, evaluate_hermite, validate_inputs,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Akima 1D interpolator.
///
/// A locally-weighted cubic interpolator that is robust to outliers.
///
/// # Example
///
/// ```ignore
/// use solvr::interpolate::Akima1DInterpolator;
///
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
/// let y = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0, 0.0], &[5], &device);
///
/// let akima = Akima1DInterpolator::new(&client, &x, &y)?;
/// let y_new = akima.evaluate(&client, &x_new)?;
/// ```
pub struct Akima1DInterpolator<R: Runtime> {
    /// X coordinates (knots). Kept for potential future GPU operations.
    #[allow(dead_code)]
    x: Tensor<R>,
    /// Y values at knots.
    y_data: Vec<f64>,
    /// Computed slopes at each knot.
    slopes: Vec<f64>,
    /// Cached x data for evaluation.
    x_data: Vec<f64>,
    /// Number of data points.
    n: usize,
    /// Minimum x value.
    x_min: f64,
    /// Maximum x value.
    x_max: f64,
}

impl<R: Runtime> Akima1DInterpolator<R> {
    /// Create a new Akima interpolator.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client for tensor operations
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
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
    ) -> InterpolateResult<Self> {
        let validated = validate_inputs(x, y, "Akima1DInterpolator::new")?;

        // Compute slopes using Akima method
        let slopes = Self::compute_slopes(&validated.x_data, &validated.y_data);

        Ok(Self {
            x: x.clone(),
            y_data: validated.y_data,
            slopes,
            x_data: validated.x_data,
            n: validated.n,
            x_min: validated.x_min,
            x_max: validated.x_max,
        })
    }

    /// Compute slopes using the Akima method.
    ///
    /// The Akima method uses weights based on the absolute differences between
    /// adjacent slopes to reduce sensitivity to outliers.
    fn compute_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut slopes = vec![0.0; n];

        if n == 2 {
            let secant = (y[1] - y[0]) / (x[1] - x[0]);
            slopes[0] = secant;
            slopes[1] = secant;
            return slopes;
        }

        // Compute secants (slopes between adjacent points)
        let mut m = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            m.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
        }

        // Extend slopes at boundaries using parabolic extrapolation
        let m_minus2 = 3.0 * m[0] - 2.0 * m[1];
        let m_minus1 = 2.0 * m[0] - m[1];
        let m_n = 2.0 * m[n - 2] - m[n - 3];
        let m_n_plus1 = 3.0 * m[n - 2] - 2.0 * m[n - 3];

        // Build extended m array
        let mut m_ext = Vec::with_capacity(n + 3);
        m_ext.push(m_minus2);
        m_ext.push(m_minus1);
        m_ext.extend_from_slice(&m);
        m_ext.push(m_n);
        m_ext.push(m_n_plus1);

        // Compute slopes at each point using Akima formula
        for (i, slope) in slopes.iter_mut().enumerate() {
            let idx = i + 2;

            let dm1 = (m_ext[idx + 1] - m_ext[idx]).abs();
            let dm2 = (m_ext[idx - 1] - m_ext[idx - 2]).abs();

            *slope = if dm1 + dm2 < 1e-14 {
                0.5 * (m_ext[idx - 1] + m_ext[idx])
            } else {
                (dm1 * m_ext[idx - 1] + dm2 * m_ext[idx]) / (dm1 + dm2)
            };
        }

        slopes
    }

    /// Evaluate the interpolant at new x coordinates.
    pub fn evaluate<C: RuntimeClient<R>>(
        &self,
        client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let data = HermiteData {
            x_data: &self.x_data,
            y_data: &self.y_data,
            slopes: &self.slopes,
            n: self.n,
            x_min: self.x_min,
            x_max: self.x_max,
            context: "Akima1DInterpolator::evaluate",
        };
        evaluate_hermite(client, x_new, &data)
    }

    /// Evaluate the first derivative at new x coordinates.
    pub fn derivative<C: RuntimeClient<R>>(
        &self,
        client: &C,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let data = HermiteData {
            x_data: &self.x_data,
            y_data: &self.y_data,
            slopes: &self.slopes,
            n: self.n,
            x_min: self.x_min,
            x_max: self.x_max,
            context: "Akima1DInterpolator::derivative",
        };
        derivative_hermite(client, x_new, &data)
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
    use crate::interpolate::error::InterpolateError;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_akima_passes_through_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 1.5, 0.5], &[5], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();
        let y_test = akima.evaluate(&client, &x).unwrap();
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

    #[test]
    fn test_akima_linear_data() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0, 9.0], &[5], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5, 3.5], &[4], &device);
        let y_test = akima.evaluate(&client, &x_test).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 2.0).abs() < 1e-10);
        assert!((y_result[1] - 4.0).abs() < 1e-10);
        assert!((y_result[2] - 6.0).abs() < 1e-10);
        assert!((y_result[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_akima_outlier_robustness() {
        let (device, client) = setup();

        let x =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[7], &device);
        let y =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 10.0, 1.0, 2.0, 3.0, 4.0], &[7], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[4.5, 5.5], &[2], &device);
        let y_test = akima.evaluate(&client, &x_test).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!(y_result[0] > 2.0 && y_result[0] < 3.5);
        assert!(y_result[1] > 3.0 && y_result[1] < 4.0);
    }

    #[test]
    fn test_akima_derivative() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 3.0, 6.0, 9.0, 12.0], &[5], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5, 3.5], &[4], &device);
        let dy = akima.derivative(&client, &x_test).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        for val in dy_result {
            assert!(
                (val - 3.0).abs() < 1e-10,
                "Derivative should be 3, got {}",
                val
            );
        }
    }

    #[test]
    fn test_akima_smooth_data() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6], &device);
        let y =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.84, 0.91, 0.14, -0.76, -0.96], &[6], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_fine: Vec<f64> = (0..51).map(|i| i as f64 * 0.1).collect();
        let x_fine_tensor = Tensor::<CpuRuntime>::from_slice(&x_fine, &[x_fine.len()], &device);
        let y_fine = akima.evaluate(&client, &x_fine_tensor).unwrap();
        let y_fine_result: Vec<f64> = y_fine.to_vec();

        for &val in &y_fine_result {
            assert!(
                (-1.5..=1.5).contains(&val),
                "Value {} outside reasonable range",
                val
            );
        }
    }

    #[test]
    fn test_akima_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1], &device);
        let result = akima.evaluate(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    #[test]
    fn test_akima_non_monotonic_x_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);

        let result = Akima1DInterpolator::new(&client, &x, &y);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_akima_shape_mismatch_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = Akima1DInterpolator::new(&client, &x, &y);
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_akima_insufficient_data_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = Akima1DInterpolator::new(&client, &x, &y);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_akima_bounds_and_len() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 5.0, 10.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        assert_eq!(akima.len(), 4);
        assert!(!akima.is_empty());
        let (min, max) = akima.bounds();
        assert!((min - 1.0).abs() < 1e-14);
        assert!((max - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_akima_two_points() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0], &[2], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let y_test = akima.evaluate(&client, &x_test).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);
        assert!((y_result[1] - 1.0).abs() < 1e-10);
        assert!((y_result[2] - 2.0).abs() < 1e-10);
    }
}
