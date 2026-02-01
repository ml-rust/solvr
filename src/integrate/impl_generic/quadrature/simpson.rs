//! Simpson's rule integration using tensor operations.
//!
//! All implementations use numr tensor ops - no scalar loops.

use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Simpson's rule integration.
///
/// Uses Simpson's 1/3 rule for even intervals, with trapezoidal for odd.
/// All operations use tensor ops for GPU acceleration.
pub fn simpson_impl<R, C>(
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
            reason: "simpson: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "simpson: need at least 2 points".to_string(),
        });
    }

    if let Some(x_val) = x {
        simpson_variable_spacing(client, y, x_val, n)
    } else {
        simpson_constant_spacing(client, y, dx, n)
    }
}

/// Simpson's rule with constant spacing using tensor operations.
fn simpson_constant_spacing<R, C>(client: &C, y: &Tensor<R>, dx: f64, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();
    let last_dim = y_shape.len() - 1;

    // For 2 points, just use trapezoidal
    if n == 2 {
        let sum_y = client.sum(y, &[last_dim], false)?;
        return client.mul_scalar(&sum_y, 0.5 * dx);
    }

    // Number of intervals
    let intervals = n - 1;

    // Create Simpson's weights: [1, 4, 2, 4, 2, ..., 4, 1] for even intervals
    // or combined Simpson + trapezoidal for odd
    let weights = create_simpson_weights(n, intervals);

    // Create weight tensor with appropriate shape for broadcasting
    let weight_shape = if y_shape.len() == 1 {
        vec![n]
    } else {
        vec![1, n]
    };
    let weight_tensor = Tensor::<R>::from_slice(&weights, &weight_shape, client.device());

    // Multiply y by weights
    let weighted = client.mul(y, &weight_tensor)?;

    // Sum along last dimension
    let sum = client.sum(&weighted, &[last_dim], false)?;

    // Multiply by dx/3
    client.mul_scalar(&sum, dx / 3.0)
}

/// Create Simpson's weights for n points.
fn create_simpson_weights(n: usize, intervals: usize) -> Vec<f64> {
    let mut weights = vec![0.0; n];

    if intervals.is_multiple_of(2) {
        // Even intervals: pure Simpson's 1/3 rule
        // Pattern: 1, 4, 2, 4, 2, ..., 4, 1
        weights[0] = 1.0;
        weights[n - 1] = 1.0;
        for (i, w) in weights.iter_mut().enumerate().take(n - 1).skip(1) {
            *w = if i % 2 == 1 { 4.0 } else { 2.0 };
        }
    } else {
        // Odd intervals: Simpson's for n-1 points, then trapezoidal for last
        // Combined weights to compute in single pass

        // First apply Simpson weights for n-1 points (n-2 intervals, which is even)
        weights[0] = 1.0;
        for (i, w) in weights.iter_mut().enumerate().take(n - 2).skip(1) {
            *w = if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        // Last point of Simpson's part
        weights[n - 2] = if (n - 2) % 2 == 1 { 4.0 } else { 1.0 };

        // Add trapezoidal contribution for last interval
        // Trap: dx/2*(y[n-2] + y[n-1]) = dx/3 * (1.5*y[n-2] + 1.5*y[n-1])
        weights[n - 2] += 1.5;
        weights[n - 1] = 1.5;
    }

    weights
}

/// Simpson's rule with variable spacing using tensor operations.
///
/// Uses index_select for strided access to implement pure tensor computation.
fn simpson_variable_spacing<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: &Tensor<R>,
    n: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();
    let last_dim = y_shape.len() - 1;
    let x_last_dim = x.shape().len() - 1;

    // For 2 points, use trapezoidal
    if n == 2 {
        let x_left = x.narrow(x_last_dim as isize, 0, 1)?.contiguous();
        let x_right = x.narrow(x_last_dim as isize, 1, 1)?.contiguous();
        let dx = client.sub(&x_right, &x_left)?;

        let y_left = y.narrow(last_dim as isize, 0, 1)?.contiguous();
        let y_right = y.narrow(last_dim as isize, 1, 1)?.contiguous();
        let y_sum = client.add(&y_left, &y_right)?;

        let area = client.mul(&dx, &y_sum)?;
        let scaled = client.mul_scalar(&area, 0.5)?;
        return client.sum(&scaled, &[last_dim], false);
    }

    let intervals = n - 1;

    if intervals.is_multiple_of(2) {
        // Even intervals: pure Simpson's rule using tensor ops
        simpson_even_intervals(client, y, x, n, last_dim, x_last_dim)
    } else {
        // Odd intervals: Simpson's for n-1 points, trapezoidal for last
        simpson_odd_intervals(client, y, x, n, last_dim, x_last_dim)
    }
}

/// Simpson's rule for even number of intervals using tensor operations.
fn simpson_even_intervals<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: &Tensor<R>,
    n: usize,
    last_dim: usize,
    x_last_dim: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let num_pairs = (n - 1) / 2;

    // Create index tensors for strided access
    // For pair j: we need points at indices 2j, 2j+1, 2j+2
    let start_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j) as i64).collect();
    let mid_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j + 1) as i64).collect();
    let end_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j + 2) as i64).collect();

    let start_idx = Tensor::<R>::from_slice(&start_indices, &[num_pairs], client.device());
    let mid_idx = Tensor::<R>::from_slice(&mid_indices, &[num_pairs], client.device());
    let end_idx = Tensor::<R>::from_slice(&end_indices, &[num_pairs], client.device());

    // Extract y values at strided positions
    let y_start = client.index_select(y, last_dim, &start_idx)?;
    let y_mid = client.index_select(y, last_dim, &mid_idx)?;
    let y_end = client.index_select(y, last_dim, &end_idx)?;

    // Extract x values at start and end of each pair
    let x_start = client.index_select(x, x_last_dim, &start_idx)?;
    let x_end = client.index_select(x, x_last_dim, &end_idx)?;

    // h = (x_end - x_start) / 2
    let x_diff = client.sub(&x_end, &x_start)?;
    let h = client.mul_scalar(&x_diff, 0.5)?;

    // Simpson's formula: h/3 * (y_start + 4*y_mid + y_end)
    // = h/3 * y_start + 4*h/3 * y_mid + h/3 * y_end
    let y_mid_scaled = client.mul_scalar(&y_mid, 4.0)?;
    let y_sum = client.add(&y_start, &y_mid_scaled)?;
    let y_total = client.add(&y_sum, &y_end)?;

    // contrib = h/3 * y_total
    let h_over_3 = client.mul_scalar(&h, 1.0 / 3.0)?;
    let contrib = client.mul(&h_over_3, &y_total)?;

    // Sum contributions along last dimension
    client.sum(&contrib, &[last_dim], false)
}

/// Simpson's rule for odd number of intervals using tensor operations.
fn simpson_odd_intervals<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: &Tensor<R>,
    n: usize,
    last_dim: usize,
    x_last_dim: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Simpson's for first n-1 points (n-2 intervals, which is even)
    // Then trapezoidal for last interval

    // Simpson's part: points 0..n-1 (n-1 points, n-2 intervals)
    let simpson_n = n - 1;
    let num_pairs = (simpson_n - 1) / 2;

    // Create index tensors for Simpson's part
    let start_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j) as i64).collect();
    let mid_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j + 1) as i64).collect();
    let end_indices: Vec<i64> = (0..num_pairs).map(|j| (2 * j + 2) as i64).collect();

    let start_idx = Tensor::<R>::from_slice(&start_indices, &[num_pairs], client.device());
    let mid_idx = Tensor::<R>::from_slice(&mid_indices, &[num_pairs], client.device());
    let end_idx = Tensor::<R>::from_slice(&end_indices, &[num_pairs], client.device());

    // Extract y values for Simpson's part
    let y_start = client.index_select(y, last_dim, &start_idx)?;
    let y_mid = client.index_select(y, last_dim, &mid_idx)?;
    let y_end = client.index_select(y, last_dim, &end_idx)?;

    // Extract x values
    let x_start = client.index_select(x, x_last_dim, &start_idx)?;
    let x_end = client.index_select(x, x_last_dim, &end_idx)?;

    // h = (x_end - x_start) / 2
    let x_diff = client.sub(&x_end, &x_start)?;
    let h = client.mul_scalar(&x_diff, 0.5)?;

    // Simpson's formula: h/3 * (y_start + 4*y_mid + y_end)
    let y_mid_scaled = client.mul_scalar(&y_mid, 4.0)?;
    let y_sum = client.add(&y_start, &y_mid_scaled)?;
    let y_total = client.add(&y_sum, &y_end)?;

    let h_over_3 = client.mul_scalar(&h, 1.0 / 3.0)?;
    let simpson_contrib = client.mul(&h_over_3, &y_total)?;
    let simpson_integral = client.sum(&simpson_contrib, &[last_dim], false)?;

    // Trapezoidal for last interval (n-2, n-1)
    let last_x_left = x.narrow(x_last_dim as isize, n - 2, 1)?.contiguous();
    let last_x_right = x.narrow(x_last_dim as isize, n - 1, 1)?.contiguous();
    let last_dx = client.sub(&last_x_right, &last_x_left)?;

    let last_y_left = y.narrow(last_dim as isize, n - 2, 1)?.contiguous();
    let last_y_right = y.narrow(last_dim as isize, n - 1, 1)?.contiguous();
    let last_y_sum = client.add(&last_y_left, &last_y_right)?;

    let trap_area = client.mul(&last_dx, &last_y_sum)?;
    let trap_scaled = client.mul_scalar(&trap_area, 0.5)?;
    let trap_integral = client.sum(&trap_scaled, &[last_dim], false)?;

    // Total = Simpson's + Trapezoidal
    client.add(&simpson_integral, &trap_integral)
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
    fn test_simpson_constant_spacing() {
        let client = get_client();

        // Integrate y = x^2 from 0 to 2 with 5 points (4 intervals)
        // Points: [0, 0.5, 1.0, 1.5, 2.0]
        // y values: [0, 0.25, 1.0, 2.25, 4.0]
        // Exact integral = 8/3 ≈ 2.6667
        let y = Tensor::from_slice(&[0.0, 0.25, 1.0, 2.25, 4.0], &[5], client.device());
        let result = simpson_impl(&client, &y, None, 0.5).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_variable_spacing_even() {
        let client = get_client();

        // Integrate y = x^2 from 0 to 2 with 5 points (4 intervals = even)
        let x = Tensor::from_slice(&[0.0, 0.5, 1.0, 1.5, 2.0], &[5], client.device());
        let y = Tensor::from_slice(&[0.0, 0.25, 1.0, 2.25, 4.0], &[5], client.device());
        let result = simpson_impl(&client, &y, Some(&x), 0.5).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_variable_spacing_odd() {
        let client = get_client();

        // Integrate y = x from 0 to 3 with 4 points (3 intervals = odd)
        // Points: [0, 1, 2, 3]
        // y values: [0, 1, 2, 3]
        // Exact integral = 4.5
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], client.device());
        let y = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], client.device());
        let result = simpson_impl(&client, &y, Some(&x), 1.0).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_batch() {
        let client = get_client();

        // Batch integration
        // Row 0: y = x, integral over [0,2] = 2
        // Row 1: y = x^2, integral over [0,2] = 8/3
        let x = Tensor::from_slice(&[0.0, 0.5, 1.0, 1.5, 2.0], &[5], client.device());
        let y = Tensor::from_slice(
            &[
                0.0, 0.5, 1.0, 1.5, 2.0, // Row 0: y = x
                0.0, 0.25, 1.0, 2.25, 4.0, // Row 1: y = x^2
            ],
            &[2, 5],
            client.device(),
        );
        let result = simpson_impl(&client, &y, Some(&x), 0.5).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 2.0).abs() < 1e-10);
        assert!((values[1] - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_two_points() {
        let client = get_client();

        // Two points should use trapezoidal
        let x = Tensor::from_slice(&[0.0, 1.0], &[2], client.device());
        let y = Tensor::from_slice(&[0.0, 1.0], &[2], client.device());
        let result = simpson_impl(&client, &y, Some(&x), 1.0).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 0.5).abs() < 1e-10);
    }
}
