//! Shared Hermite cubic interpolation primitives using tensor operations.
//!
//! This module provides the common building blocks for Hermite-based interpolators
//! (PCHIP, Akima, etc.). All operations use tensor ops - data stays on device.
//!
//! # Hermite Cubic Interpolation
//!
//! Given values y0, y1 and slopes d0, d1 at interval endpoints x0, x1,
//! the Hermite cubic polynomial is:
//!
//! ```text
//! p(x) = h00(t)*y0 + h10(t)*h*d0 + h01(t)*y1 + h11(t)*h*d1
//!
//! where t = (x - x0) / h, h = x1 - x0
//!
//! h00(t) = 2t³ - 3t² + 1
//! h10(t) = t³ - 2t² + t
//! h01(t) = -2t³ + 3t²
//! h11(t) = t³ - t²
//! ```

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validated input data from 1D interpolation tensors.
pub struct ValidatedData<R: Runtime> {
    /// X coordinates as tensor (stays on device).
    pub x: Tensor<R>,
    /// Y values as tensor (stays on device).
    pub y: Tensor<R>,
    /// Number of data points.
    pub n: usize,
    /// Minimum x value (scalar for bounds checking).
    pub x_min: f64,
    /// Maximum x value (scalar for bounds checking).
    pub x_max: f64,
}

/// Data required for Hermite interpolation evaluation.
pub struct HermiteDataTensor<'a, R: Runtime> {
    /// X coordinates (knots).
    pub x: &'a Tensor<R>,
    /// Y values at knots.
    pub y: &'a Tensor<R>,
    /// Slopes at each knot.
    pub slopes: &'a Tensor<R>,
    /// Number of data points.
    pub n: usize,
}

/// Validate input tensors for 1D interpolation.
///
/// Checks that:
/// - x and y are 1D tensors
/// - x and y have the same length
/// - At least 2 data points are provided
/// - x values are strictly increasing
pub fn validate_inputs<R: Runtime>(
    x: &Tensor<R>,
    y: &Tensor<R>,
    context: &str,
) -> InterpolateResult<ValidatedData<R>> {
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
            context: context.to_string(),
        });
    }

    if n < 2 {
        return Err(InterpolateError::InsufficientData {
            required: 2,
            actual: n,
            context: context.to_string(),
        });
    }

    // Extract min/max for bounds checking (small transfer, done once)
    let x_data: Vec<f64> = x.contiguous().to_vec();

    // Check strictly increasing
    for i in 1..n {
        if x_data[i] <= x_data[i - 1] {
            return Err(InterpolateError::NotMonotonic {
                context: context.to_string(),
            });
        }
    }

    let x_min = x_data[0];
    let x_max = x_data[n - 1];

    Ok(ValidatedData {
        x: x.clone(),
        y: y.clone(),
        n,
        x_min,
        x_max,
    })
}

/// Evaluate a Hermite interpolant at multiple points using tensor operations.
///
/// All computation stays on device. Uses searchsorted for interval finding
/// and gather for coefficient lookup.
pub fn evaluate_hermite_tensor<R, C>(
    client: &C,
    x_new: &Tensor<R>,
    data: &HermiteDataTensor<'_, R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let x_new_shape = x_new.shape();
    if x_new_shape.len() != 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "x_new".to_string(),
            message: "x_new must be a 1D tensor".to_string(),
        });
    }

    // Find intervals using searchsorted (stays on device)
    // Out-of-bounds queries are clamped to boundary intervals
    // searchsorted gives insertion points (I64), we need interval indices
    let indices = client
        .searchsorted(data.x, x_new, false)
        .map_err(to_interp_err)?;

    // Clamp indices to [1, n-1] then subtract 1 to get interval [0, n-2]
    // indices = clamp(indices, 1, n-1) - 1
    let n = data.n;
    let ones_i64 = create_constant_tensor_i64(client, x_new_shape[0], 1)?;
    let n_minus_1_i64 = create_constant_tensor_i64(client, x_new_shape[0], (n - 1) as i64)?;

    // Clamp: max(1, min(indices, n-1))
    let indices_clamped = client
        .maximum(
            &client
                .minimum(&indices, &n_minus_1_i64)
                .map_err(to_interp_err)?,
            &ones_i64,
        )
        .map_err(to_interp_err)?;
    let idx = client
        .sub(&indices_clamped, &ones_i64)
        .map_err(to_interp_err)?;

    // Gather values at interval endpoints
    // x0 = x[idx], x1 = x[idx+1], y0 = y[idx], y1 = y[idx+1], d0 = slopes[idx], d1 = slopes[idx+1]
    let idx_plus_1 = client.add(&idx, &ones_i64).map_err(to_interp_err)?;

    let x0 = client
        .index_select(data.x, 0, &idx)
        .map_err(to_interp_err)?;
    let x1 = client
        .index_select(data.x, 0, &idx_plus_1)
        .map_err(to_interp_err)?;
    let y0 = client
        .index_select(data.y, 0, &idx)
        .map_err(to_interp_err)?;
    let y1 = client
        .index_select(data.y, 0, &idx_plus_1)
        .map_err(to_interp_err)?;
    let d0 = client
        .index_select(data.slopes, 0, &idx)
        .map_err(to_interp_err)?;
    let d1 = client
        .index_select(data.slopes, 0, &idx_plus_1)
        .map_err(to_interp_err)?;

    // Compute Hermite polynomial
    // h = x1 - x0
    // t = (x_new - x0) / h
    let h = client.sub(&x1, &x0).map_err(to_interp_err)?;
    let x_shifted = client.sub(x_new, &x0).map_err(to_interp_err)?;
    let t = client.div(&x_shifted, &h).map_err(to_interp_err)?;

    // t², t³
    let t2 = client.mul(&t, &t).map_err(to_interp_err)?;
    let t3 = client.mul(&t2, &t).map_err(to_interp_err)?;

    // Hermite basis functions:
    // h00 = 2t³ - 3t² + 1
    // h10 = t³ - 2t² + t
    // h01 = -2t³ + 3t²
    // h11 = t³ - t²

    // h00 = 2*t3 - 3*t2 + 1
    let h00 = client
        .add_scalar(
            &client
                .sub(
                    &client.mul_scalar(&t3, 2.0).map_err(to_interp_err)?,
                    &client.mul_scalar(&t2, 3.0).map_err(to_interp_err)?,
                )
                .map_err(to_interp_err)?,
            1.0,
        )
        .map_err(to_interp_err)?;

    // h10 = t3 - 2*t2 + t
    let h10 = client
        .add(
            &client
                .sub(&t3, &client.mul_scalar(&t2, 2.0).map_err(to_interp_err)?)
                .map_err(to_interp_err)?,
            &t,
        )
        .map_err(to_interp_err)?;

    // h01 = -2*t3 + 3*t2
    let h01 = client
        .add(
            &client.mul_scalar(&t3, -2.0).map_err(to_interp_err)?,
            &client.mul_scalar(&t2, 3.0).map_err(to_interp_err)?,
        )
        .map_err(to_interp_err)?;

    // h11 = t3 - t2
    let h11 = client.sub(&t3, &t2).map_err(to_interp_err)?;

    // result = h00*y0 + h10*h*d0 + h01*y1 + h11*h*d1
    let term1 = client.mul(&h00, &y0).map_err(to_interp_err)?;
    let term2 = client
        .mul(&h10, &client.mul(&h, &d0).map_err(to_interp_err)?)
        .map_err(to_interp_err)?;
    let term3 = client.mul(&h01, &y1).map_err(to_interp_err)?;
    let term4 = client
        .mul(&h11, &client.mul(&h, &d1).map_err(to_interp_err)?)
        .map_err(to_interp_err)?;

    let result = client
        .add(
            &client.add(&term1, &term2).map_err(to_interp_err)?,
            &client.add(&term3, &term4).map_err(to_interp_err)?,
        )
        .map_err(to_interp_err)?;

    Ok(result)
}

/// Evaluate the derivative of a Hermite interpolant using tensor operations.
pub fn derivative_hermite_tensor<R, C>(
    client: &C,
    x_new: &Tensor<R>,
    data: &HermiteDataTensor<'_, R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let x_new_shape = x_new.shape();
    if x_new_shape.len() != 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "x_new".to_string(),
            message: "x_new must be a 1D tensor".to_string(),
        });
    }

    // Find intervals (out-of-bounds queries clamped to boundary)
    let indices = client
        .searchsorted(data.x, x_new, false)
        .map_err(to_interp_err)?;

    let n = data.n;
    let ones_i64 = create_constant_tensor_i64(client, x_new_shape[0], 1)?;
    let n_minus_1_i64 = create_constant_tensor_i64(client, x_new_shape[0], (n - 1) as i64)?;

    let indices_clamped = client
        .maximum(
            &client
                .minimum(&indices, &n_minus_1_i64)
                .map_err(to_interp_err)?,
            &ones_i64,
        )
        .map_err(to_interp_err)?;
    let idx = client
        .sub(&indices_clamped, &ones_i64)
        .map_err(to_interp_err)?;
    let idx_plus_1 = client.add(&idx, &ones_i64).map_err(to_interp_err)?;

    // Gather values
    let x0 = client
        .index_select(data.x, 0, &idx)
        .map_err(to_interp_err)?;
    let x1 = client
        .index_select(data.x, 0, &idx_plus_1)
        .map_err(to_interp_err)?;
    let y0 = client
        .index_select(data.y, 0, &idx)
        .map_err(to_interp_err)?;
    let y1 = client
        .index_select(data.y, 0, &idx_plus_1)
        .map_err(to_interp_err)?;
    let d0 = client
        .index_select(data.slopes, 0, &idx)
        .map_err(to_interp_err)?;
    let d1 = client
        .index_select(data.slopes, 0, &idx_plus_1)
        .map_err(to_interp_err)?;

    // Compute derivative of Hermite polynomial
    let h = client.sub(&x1, &x0).map_err(to_interp_err)?;
    let x_shifted = client.sub(x_new, &x0).map_err(to_interp_err)?;
    let t = client.div(&x_shifted, &h).map_err(to_interp_err)?;
    let t2 = client.mul(&t, &t).map_err(to_interp_err)?;

    // Derivatives of Hermite basis functions (with chain rule factor 1/h):
    // dh00/dx = (6t² - 6t) / h
    // dh10/dx = 3t² - 4t + 1
    // dh01/dx = (-6t² + 6t) / h
    // dh11/dx = 3t² - 2t

    // dh00 = (6*t2 - 6*t) / h
    let dh00 = client
        .div(
            &client
                .sub(
                    &client.mul_scalar(&t2, 6.0).map_err(to_interp_err)?,
                    &client.mul_scalar(&t, 6.0).map_err(to_interp_err)?,
                )
                .map_err(to_interp_err)?,
            &h,
        )
        .map_err(to_interp_err)?;

    // dh10 = 3*t2 - 4*t + 1
    let dh10 = client
        .add_scalar(
            &client
                .sub(
                    &client.mul_scalar(&t2, 3.0).map_err(to_interp_err)?,
                    &client.mul_scalar(&t, 4.0).map_err(to_interp_err)?,
                )
                .map_err(to_interp_err)?,
            1.0,
        )
        .map_err(to_interp_err)?;

    // dh01 = (-6*t2 + 6*t) / h
    let dh01 = client
        .div(
            &client
                .add(
                    &client.mul_scalar(&t2, -6.0).map_err(to_interp_err)?,
                    &client.mul_scalar(&t, 6.0).map_err(to_interp_err)?,
                )
                .map_err(to_interp_err)?,
            &h,
        )
        .map_err(to_interp_err)?;

    // dh11 = 3*t2 - 2*t
    let dh11 = client
        .sub(
            &client.mul_scalar(&t2, 3.0).map_err(to_interp_err)?,
            &client.mul_scalar(&t, 2.0).map_err(to_interp_err)?,
        )
        .map_err(to_interp_err)?;

    // result = dh00*y0 + dh10*d0 + dh01*y1 + dh11*d1
    let term1 = client.mul(&dh00, &y0).map_err(to_interp_err)?;
    let term2 = client.mul(&dh10, &d0).map_err(to_interp_err)?;
    let term3 = client.mul(&dh01, &y1).map_err(to_interp_err)?;
    let term4 = client.mul(&dh11, &d1).map_err(to_interp_err)?;

    let result = client
        .add(
            &client.add(&term1, &term2).map_err(to_interp_err)?,
            &client.add(&term3, &term4).map_err(to_interp_err)?,
        )
        .map_err(to_interp_err)?;

    Ok(result)
}

/// Create a constant tensor with the given value.
fn create_constant_tensor_i64<R, C>(
    client: &C,
    len: usize,
    value: i64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    // Create constant I64 tensor for index operations
    let data = vec![value; len];
    Ok(Tensor::from_slice(&data, &[len], client.device()))
}

/// Convert numr error to interpolation error.
fn to_interp_err(e: numr::error::Error) -> InterpolateError {
    InterpolateError::NumericalError {
        message: format!("Tensor operation failed: {}", e),
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
    fn test_validate_inputs() {
        let (device, _client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0], &[3], &device);

        let result = validate_inputs(&x, &y, "test");
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.n, 3);
        assert!((data.x_min - 0.0).abs() < 1e-10);
        assert!((data.x_max - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_inputs_non_monotonic() {
        let (device, _client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = validate_inputs(&x, &y, "test");
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_validate_inputs_shape_mismatch() {
        let (device, _client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = validate_inputs(&x, &y, "test");
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }
}
