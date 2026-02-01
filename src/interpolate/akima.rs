//! Akima spline interpolation using tensor operations.
//!
//! Akima spline is a piecewise cubic interpolation method that:
//! - Is less sensitive to outliers than traditional cubic splines
//! - Has continuous first derivative (C1)
//! - Uses locally-weighted slopes to reduce oscillation
//!
//! All computation uses tensor ops - data stays on device.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::hermite_core::{
    HermiteDataTensor, derivative_hermite_tensor, evaluate_hermite_tensor, validate_inputs,
};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Akima 1D interpolator using tensor operations.
///
/// A locally-weighted cubic interpolator that is robust to outliers.
/// All data stays on device as tensors.
pub struct Akima1DInterpolator<R: Runtime> {
    /// X coordinates (knots).
    x: Tensor<R>,
    /// Y values at knots.
    y: Tensor<R>,
    /// Computed slopes at each knot.
    slopes: Tensor<R>,
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
    pub fn new<C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Self>
    where
        C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    {
        let validated = validate_inputs(x, y, "Akima1DInterpolator::new")?;

        // Compute slopes using Akima method with tensor ops
        let slopes = compute_akima_slopes(client, &validated.x, &validated.y)?;

        Ok(Self {
            x: validated.x,
            y: validated.y,
            slopes,
            n: validated.n,
            x_min: validated.x_min,
            x_max: validated.x_max,
        })
    }

    /// Evaluate the interpolant at new x coordinates.
    pub fn evaluate<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    {
        let data = HermiteDataTensor {
            x: &self.x,
            y: &self.y,
            slopes: &self.slopes,
            n: self.n,
        };
        evaluate_hermite_tensor(client, x_new, &data)
    }

    /// Evaluate the first derivative at new x coordinates.
    pub fn derivative<C>(&self, client: &C, x_new: &Tensor<R>) -> InterpolateResult<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
    {
        let data = HermiteDataTensor {
            x: &self.x,
            y: &self.y,
            slopes: &self.slopes,
            n: self.n,
        };
        derivative_hermite_tensor(client, x_new, &data)
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

/// Compute slopes using the Akima method with pure tensor operations.
///
/// The Akima method uses weights based on the absolute differences between
/// adjacent slopes to reduce sensitivity to outliers.
///
/// All computation stays on device - no to_vec() calls.
fn compute_akima_slopes<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];
    let device = client.device();

    if n == 2 {
        // For 2 points, slope is just the secant: (y[1] - y[0]) / (x[1] - x[0])
        // Use narrow to slice and compute
        let x0 = x.narrow(0, 0, 1)?;
        let x1 = x.narrow(0, 1, 1)?;
        let y0 = y.narrow(0, 0, 1)?;
        let y1 = y.narrow(0, 1, 1)?;
        let dx = client.sub(&x1, &x0)?;
        let dy = client.sub(&y1, &y0)?;
        let secant = client.div(&dy, &dx)?;
        // Return [secant, secant] - cat two copies
        return Ok(client.cat(&[&secant, &secant], 0)?);
    }

    // Step 1: Compute secants between adjacent points using tensor slicing
    // m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) for i = 0..n-2
    let x_lo = x.narrow(0, 0, n - 1)?; // x[0..n-1]
    let x_hi = x.narrow(0, 1, n - 1)?; // x[1..n]
    let y_lo = y.narrow(0, 0, n - 1)?; // y[0..n-1]
    let y_hi = y.narrow(0, 1, n - 1)?; // y[1..n]

    let dx = client.sub(&x_hi, &x_lo)?;
    let dy = client.sub(&y_hi, &y_lo)?;
    let m = client.div(&dy, &dx)?; // Shape: [n-1]

    // Step 2: Extend slopes at boundaries using parabolic extrapolation
    // m[-2] = 3*m[0] - 2*m[1]
    // m[-1] = 2*m[0] - m[1]
    // m[n-1] = 2*m[n-2] - m[n-3]
    // m[n] = 3*m[n-2] - 2*m[n-3]
    let m0 = m.narrow(0, 0, 1)?; // m[0]
    let m1 = m.narrow(0, 1, 1)?; // m[1]
    let m_last = m.narrow(0, n - 2, 1)?; // m[n-2] (last element)
    let m_second_last = m.narrow(0, n - 3, 1)?; // m[n-3] (second to last)

    // m_minus2 = 3*m[0] - 2*m[1]
    let m0_3 = client.mul_scalar(&m0, 3.0)?;
    let m1_2 = client.mul_scalar(&m1, 2.0)?;
    let m_minus2 = client.sub(&m0_3, &m1_2)?;

    // m_minus1 = 2*m[0] - m[1]
    let m0_2 = client.mul_scalar(&m0, 2.0)?;
    let m_minus1 = client.sub(&m0_2, &m1)?;

    // m_n = 2*m[n-2] - m[n-3]
    let m_last_2 = client.mul_scalar(&m_last, 2.0)?;
    let m_n = client.sub(&m_last_2, &m_second_last)?;

    // m_n_plus1 = 3*m[n-2] - 2*m[n-3]
    let m_last_3 = client.mul_scalar(&m_last, 3.0)?;
    let m_second_last_2 = client.mul_scalar(&m_second_last, 2.0)?;
    let m_n_plus1 = client.sub(&m_last_3, &m_second_last_2)?;

    // Build extended m array: [m_minus2, m_minus1, m[0..n-1], m_n, m_n_plus1]
    // Shape: [n+3]
    let m_ext = client.cat(&[&m_minus2, &m_minus1, &m, &m_n, &m_n_plus1], 0)?;

    // Step 3: Compute slopes at each point using vectorized Akima formula
    // For slope[i] (i = 0..n), we use m_ext indices:
    //   m_ext[i] = m[-2+i], m_ext[i+1] = m[-1+i], m_ext[i+2] = m[i], m_ext[i+3] = m[i+1]
    // So:
    //   dm1 = |m_ext[i+3] - m_ext[i+2]| = |m[i+1] - m[i]|
    //   dm2 = |m_ext[i+1] - m_ext[i]| = |m[i-1] - m[i-2]|
    //   slope = (dm1 * m_ext[i+1] + dm2 * m_ext[i+2]) / (dm1 + dm2)
    //         = (dm1 * m[i-1] + dm2 * m[i]) / (dm1 + dm2)
    // With fallback: if dm1 + dm2 < eps, slope = 0.5 * (m[i-1] + m[i])

    // Slice m_ext into 4 views of length n
    let m_i_minus2 = m_ext.narrow(0, 0, n)?; // m_ext[0..n] = m[-2..-2+n]
    let m_i_minus1 = m_ext.narrow(0, 1, n)?; // m_ext[1..n+1] = m[-1..-1+n]
    let m_i = m_ext.narrow(0, 2, n)?; // m_ext[2..n+2] = m[0..n]
    let m_i_plus1 = m_ext.narrow(0, 3, n)?; // m_ext[3..n+3] = m[1..n+1]

    // dm1 = |m[i+1] - m[i]|
    let diff1 = client.sub(&m_i_plus1, &m_i)?;
    let dm1 = client.abs(&diff1)?;

    // dm2 = |m[i-1] - m[i-2]|
    let diff2 = client.sub(&m_i_minus1, &m_i_minus2)?;
    let dm2 = client.abs(&diff2)?;

    // denom = dm1 + dm2
    let denom = client.add(&dm1, &dm2)?;

    // Simple slope (fallback): 0.5 * (m[i-1] + m[i])
    let sum_slopes = client.add(&m_i_minus1, &m_i)?;
    let slope_simple = client.mul_scalar(&sum_slopes, 0.5)?;

    // For the weighted slope, we need to avoid division by zero.
    // Add a small epsilon to the denominator to prevent NaN.
    // When denom is very small, the weighted and simple slopes converge anyway.
    let epsilon_data = vec![1e-14; n];
    let epsilon = Tensor::<R>::from_slice(&epsilon_data, &[n], device);

    // Make tensors contiguous to avoid issues with non-contiguous views
    let denom_contig = denom.contiguous();
    let dm1_contig = dm1.contiguous();
    let dm2_contig = dm2.contiguous();
    let m_i_minus1_contig = m_i_minus1.contiguous();
    let m_i_contig = m_i.contiguous();
    let slope_simple_contig = slope_simple.contiguous();

    // Safe denominator: denom + epsilon
    let safe_denom = client.add(&denom_contig, &epsilon)?;

    // Weighted slope with safe denominator: (dm1 * m[i-1] + dm2 * m[i]) / safe_denom
    let term1 = client.mul(&dm1_contig, &m_i_minus1_contig)?;
    let term2 = client.mul(&dm2_contig, &m_i_contig)?;
    let numer = client.add(&term1, &term2)?;
    let slope_weighted = client.div(&numer, &safe_denom)?;

    // Blend: when denom is large, use weighted; when small, blend towards simple
    // weight = denom / safe_denom = denom / (denom + epsilon)
    // For large denom: weight ≈ 1
    // For small denom: weight ≈ 0
    let weight = client.div(&denom_contig, &safe_denom)?;

    // one_minus_weight = 1 - weight
    let ones_data = vec![1.0; n];
    let ones = Tensor::<R>::from_slice(&ones_data, &[n], device);
    let one_minus_weight = client.sub(&ones, &weight)?;

    // slopes = slope_weighted * weight + slope_simple * (1 - weight)
    let weighted_term = client.mul(&slope_weighted, &weight)?;
    let simple_term = client.mul(&slope_simple_contig, &one_minus_weight)?;
    let slopes = client.add(&weighted_term, &simple_term)?;

    Ok(slopes)
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
    fn test_akima_out_of_bounds_clamps() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);

        let akima = Akima1DInterpolator::new(&client, &x, &y).unwrap();

        // Out-of-bounds queries are clamped to boundary intervals
        let x_oob = Tensor::<CpuRuntime>::from_slice(&[-0.5, 3.5], &[2], &device);
        let result = akima.evaluate(&client, &x_oob).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Results should be computed using the boundary intervals
        assert_eq!(result_data.len(), 2);
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
