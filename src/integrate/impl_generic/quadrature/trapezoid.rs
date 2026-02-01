//! Trapezoidal rule integration using tensor operations.
//!
//! All implementations use numr tensor ops - no scalar loops.

use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trapezoidal rule integration using tensors.
///
/// Computes ∫y dx using the composite trapezoidal rule.
/// For 1D tensors, returns a 0-D tensor with the integral.
/// For 2D tensors, integrates each row and returns a 1D tensor.
///
/// Uses tensor operations throughout - works efficiently on CPU, CUDA, and WebGPU.
pub fn trapezoid_impl<R, C>(client: &C, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();
    let x_shape = x.shape();

    if y_shape.is_empty() || x_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y/x",
            reason: "trapezoid: tensors must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];
    let x_n = x_shape[x_shape.len() - 1];

    if n != x_n {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "trapezoid: x and y must have same length in last dimension (got {} and {})",
                x_n, n
            ),
        });
    }

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid: need at least 2 points".to_string(),
        });
    }

    let last_dim = y_shape.len() - 1;
    let x_last_dim = x_shape.len() - 1;

    // Compute dx = x[1:] - x[:-1] using tensor ops
    let x_left = x.narrow(x_last_dim as isize, 0, n - 1)?.contiguous();
    let x_right = x.narrow(x_last_dim as isize, 1, n - 1)?.contiguous();
    let dx = client.sub(&x_right, &x_left)?;

    // Compute y_left = y[:-1], y_right = y[1:]
    let y_left = y.narrow(last_dim as isize, 0, n - 1)?.contiguous();
    let y_right = y.narrow(last_dim as isize, 1, n - 1)?.contiguous();

    // y_sum = y_left + y_right
    let y_sum = client.add(&y_left, &y_right)?;

    // areas = 0.5 * dx * y_sum
    let scaled_y = client.mul_scalar(&y_sum, 0.5)?;
    let areas = client.mul(&dx, &scaled_y)?;

    // Sum along last dimension to get integral
    client.sum(&areas, &[last_dim], false)
}

/// Trapezoidal rule with uniform spacing.
///
/// Uses the formula: integral = dx * (sum(y) - 0.5*(y[0] + y[n-1]))
/// All operations are tensor-based.
pub fn trapezoid_uniform_impl<R, C>(client: &C, y: &Tensor<R>, dx: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();

    if y_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid_uniform: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid_uniform: need at least 2 points".to_string(),
        });
    }

    let last_dim = y_shape.len() - 1;

    // Trapezoidal rule: integral = dx * (0.5*y[0] + y[1] + ... + y[n-2] + 0.5*y[n-1])
    //                           = dx * sum(y) - 0.5*dx*(y[0] + y[n-1])

    // Sum all y values
    let total_sum = client.sum(y, &[last_dim], false)?;

    // Get endpoints: y[0] and y[n-1]
    let y_first = y.narrow(last_dim as isize, 0, 1)?.contiguous();
    let y_last = y.narrow(last_dim as isize, n - 1, 1)?.contiguous();

    // endpoints_sum = y[0] + y[n-1], then reduce the size-1 dimension
    let endpoints = client.add(&y_first, &y_last)?;
    let endpoints_sum = client.sum(&endpoints, &[last_dim], false)?;

    // integral = dx * total_sum - 0.5 * dx * endpoints_sum
    let scaled_total = client.mul_scalar(&total_sum, dx)?;
    let endpoint_correction = client.mul_scalar(&endpoints_sum, 0.5 * dx)?;

    client.sub(&scaled_total, &endpoint_correction)
}

/// Cumulative trapezoidal integration.
///
/// Returns a tensor with n-1 elements containing cumulative integrals.
/// Uses tensor operations including cumsum for efficient GPU execution.
pub fn cumulative_trapezoid_impl<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: Option<&Tensor<R>>,
    dx: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();

    if y_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "cumulative_trapezoid: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "cumulative_trapezoid: need at least 2 points".to_string(),
        });
    }

    let last_dim = y_shape.len() - 1;

    // Compute y_left = y[:-1], y_right = y[1:]
    let y_left = y.narrow(last_dim as isize, 0, n - 1)?.contiguous();
    let y_right = y.narrow(last_dim as isize, 1, n - 1)?.contiguous();

    // y_sum = y_left + y_right
    let y_sum = client.add(&y_left, &y_right)?;

    // Compute per-interval areas
    let areas = if let Some(x_tensor) = x {
        // Variable spacing: areas[i] = 0.5 * (x[i+1] - x[i]) * (y[i] + y[i+1])
        let x_shape = x_tensor.shape();
        let x_last_dim = x_shape.len() - 1;

        let x_left = x_tensor.narrow(x_last_dim as isize, 0, n - 1)?.contiguous();
        let x_right = x_tensor.narrow(x_last_dim as isize, 1, n - 1)?.contiguous();
        let dx_tensor = client.sub(&x_right, &x_left)?;

        let scaled_y = client.mul_scalar(&y_sum, 0.5)?;
        client.mul(&dx_tensor, &scaled_y)?
    } else {
        // Uniform spacing: areas[i] = 0.5 * dx * (y[i] + y[i+1])
        client.mul_scalar(&y_sum, 0.5 * dx)?
    };

    // Cumulative sum along last dimension
    client.cumsum(&areas, last_dim as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice};

    fn get_client() -> CpuClient {
        let device = CpuDevice::new();
        CpuClient::new(device)
    }

    #[test]
    fn test_trapezoid_uniform() {
        let client = get_client();

        // Integrate y = x from 0 to 1 with 5 points
        // Points: [0, 0.25, 0.5, 0.75, 1.0]
        // Exact integral = 0.5
        let y = Tensor::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], client.device());
        let result = trapezoid_uniform_impl(&client, &y, 0.25).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trapezoid_variable() {
        let client = get_client();

        // Integrate y = x from 0 to 1
        let x = Tensor::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], client.device());
        let y = Tensor::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], client.device());
        let result = trapezoid_impl(&client, &y, &x).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trapezoid_batch() {
        let client = get_client();

        // Batch integration: two rows
        // Row 0: y = x, integral = 0.5
        // Row 1: y = 2x, integral = 1.0
        let x = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3], client.device());
        let y = Tensor::from_slice(&[0.0, 0.5, 1.0, 0.0, 1.0, 2.0], &[2, 3], client.device());
        let result = trapezoid_impl(&client, &y, &x).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 0.5).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_trapezoid() {
        let client = get_client();

        // Cumulative integral of y = 1 (constant)
        // With dx = 1: cumulative = [1, 2, 3, 4]
        let y = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0], &[5], client.device());
        let result = cumulative_trapezoid_impl(&client, &y, None, 1.0).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_trapezoid_variable() {
        let client = get_client();

        // Cumulative integral with variable spacing
        let x = Tensor::from_slice(&[0.0, 1.0, 3.0, 6.0], &[4], client.device());
        let y = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[4], client.device());
        let result = cumulative_trapezoid_impl(&client, &y, Some(&x), 1.0).unwrap();

        let values: Vec<f64> = result.to_vec();
        // Intervals: [0,1], [1,3], [3,6] with widths 1, 2, 3
        // Cumulative: 1, 1+2=3, 3+3=6
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
        assert!((values[2] - 6.0).abs() < 1e-10);
    }
}
