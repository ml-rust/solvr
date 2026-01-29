//! PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
//!
//! PCHIP is a shape-preserving interpolation method that:
//! - Passes through all data points
//! - Preserves monotonicity of the data (no overshoot)
//! - Has continuous first derivative
//! - Uses Fritsch-Carlson algorithm for slope calculation
//!
//! # When to Use PCHIP vs CubicSpline
//!
//! | Property          | PCHIP                | CubicSpline          |
//! |-------------------|----------------------|----------------------|
//! | Smoothness        | C1 (continuous 1st)  | C2 (continuous 2nd)  |
//! | Monotonicity      | Preserved            | Not preserved        |
//! | Overshoot         | None                 | Possible             |
//! | Best for          | Monotonic data       | Smooth curves        |
//!
//! # Algorithm
//!
//! PCHIP uses the Fritsch-Carlson method:
//! 1. Compute slopes using secants of adjacent intervals
//! 2. At each interior point, if secants have same sign, use harmonic mean
//! 3. If secants have opposite signs or either is zero, set slope to zero
//! 4. Use Hermite basis functions to construct cubic polynomial

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::hermite_core::{
    HermiteData, derivative_hermite, evaluate_hermite, validate_inputs,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.
///
/// A monotonicity-preserving cubic interpolator that avoids overshooting.
///
/// # Example
///
/// ```ignore
/// use solvr::interpolate::PchipInterpolator;
///
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
/// let y = Tensor::from_slice(&[0.0, 0.5, 0.8, 1.0], &[4], &device);  // monotonic
///
/// let pchip = PchipInterpolator::new(&client, &x, &y)?;
/// let y_new = pchip.evaluate(&client, &x_new)?;
/// ```
pub struct PchipInterpolator<R: Runtime> {
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

impl<R: Runtime> PchipInterpolator<R> {
    /// Create a new PCHIP interpolator.
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
        let validated = validate_inputs(x, y, "PchipInterpolator::new")?;

        // Compute slopes using Fritsch-Carlson method
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

    /// Compute slopes using the Fritsch-Carlson method.
    ///
    /// This method ensures monotonicity preservation:
    /// - If adjacent secants have the same sign, use weighted harmonic mean
    /// - If they have opposite signs or either is zero, set slope to zero
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
        let mut secants = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            secants.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
        }

        // Compute interior slopes using Fritsch-Carlson
        for i in 1..n - 1 {
            let s0 = secants[i - 1];
            let s1 = secants[i];

            if s0 * s1 <= 0.0 {
                slopes[i] = 0.0;
            } else {
                let h0 = x[i] - x[i - 1];
                let h1 = x[i + 1] - x[i];
                let w1 = 2.0 * h1 + h0;
                let w2 = h1 + 2.0 * h0;
                slopes[i] = (w1 + w2) / (w1 / s0 + w2 / s1);
            }
        }

        // Endpoint slopes using one-sided differences with shape preservation
        slopes[0] = Self::endpoint_slope(secants[0], secants[1], x[1] - x[0], x[2] - x[1]);
        slopes[n - 1] = Self::endpoint_slope(
            secants[n - 2],
            secants[n - 3],
            x[n - 1] - x[n - 2],
            x[n - 2] - x[n - 3],
        );

        slopes
    }

    /// Compute endpoint slope with shape preservation.
    fn endpoint_slope(s1: f64, s2: f64, h1: f64, h2: f64) -> f64 {
        let d = ((2.0 * h1 + h2) * s1 - h1 * s2) / (h1 + h2);

        if d.signum() != s1.signum() {
            0.0
        } else if s1.signum() != s2.signum() && d.abs() > 3.0 * s1.abs() {
            3.0 * s1
        } else {
            d
        }
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
            context: "PchipInterpolator::evaluate",
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
            context: "PchipInterpolator::derivative",
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
    fn test_pchip_passes_through_knots() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 0.8, 1.0], &[4], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();
        let y_test = pchip.evaluate(&client, &x).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 0.0).abs() < 1e-10);
        assert!((y_result[1] - 0.5).abs() < 1e-10);
        assert!((y_result[2] - 0.8).abs() < 1e-10);
        assert!((y_result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pchip_preserves_monotonicity() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.2, 0.5, 0.8, 1.0], &[5], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        let x_fine: Vec<f64> = (0..41).map(|i| i as f64 * 0.1).collect();
        let x_fine_tensor = Tensor::<CpuRuntime>::from_slice(&x_fine, &[x_fine.len()], &device);
        let y_fine = pchip.evaluate(&client, &x_fine_tensor).unwrap();
        let y_fine_result: Vec<f64> = y_fine.to_vec();

        for i in 1..y_fine_result.len() {
            assert!(
                y_fine_result[i] >= y_fine_result[i - 1] - 1e-10,
                "Monotonicity violated at i={}: {} < {}",
                i,
                y_fine_result[i],
                y_fine_result[i - 1]
            );
        }
    }

    #[test]
    fn test_pchip_no_overshoot() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        let x_fine: Vec<f64> = (0..31).map(|i| i as f64 * 0.1).collect();
        let x_fine_tensor = Tensor::<CpuRuntime>::from_slice(&x_fine, &[x_fine.len()], &device);
        let y_fine = pchip.evaluate(&client, &x_fine_tensor).unwrap();
        let y_fine_result: Vec<f64> = y_fine.to_vec();

        for (i, &val) in y_fine_result.iter().enumerate() {
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&val),
                "Overshoot at i={}: value={} outside [0, 1]",
                i,
                val
            );
        }
    }

    #[test]
    fn test_pchip_linear_data() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0], &[4], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let y_test = pchip.evaluate(&client, &x_test).unwrap();
        let y_result: Vec<f64> = y_test.to_vec();

        assert!((y_result[0] - 2.0).abs() < 1e-10);
        assert!((y_result[1] - 4.0).abs() < 1e-10);
        assert!((y_result[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_pchip_derivative() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        let x_test = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let dy = pchip.derivative(&client, &x_test).unwrap();
        let dy_result: Vec<f64> = dy.to_vec();

        for val in dy_result {
            assert!(
                (val - 2.0).abs() < 1e-10,
                "Derivative should be 2, got {}",
                val
            );
        }
    }

    #[test]
    fn test_pchip_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1], &device);
        let result = pchip.evaluate(&client, &x_oob);
        assert!(matches!(result, Err(InterpolateError::OutOfDomain { .. })));
    }

    #[test]
    fn test_pchip_non_monotonic_x_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = PchipInterpolator::new(&client, &x, &y);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_pchip_shape_mismatch_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = PchipInterpolator::new(&client, &x, &y);
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_pchip_insufficient_data_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = PchipInterpolator::new(&client, &x, &y);
        assert!(matches!(
            result,
            Err(InterpolateError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_pchip_bounds_and_len() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 5.0, 10.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let pchip = PchipInterpolator::new(&client, &x, &y).unwrap();

        assert_eq!(pchip.len(), 4);
        assert!(!pchip.is_empty());
        let (min, max) = pchip.bounds();
        assert!((min - 1.0).abs() < 1e-14);
        assert!((max - 10.0).abs() < 1e-14);
    }
}
