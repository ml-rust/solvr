//! Simpson's rule integration.

use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Simpson's rule integration.
///
/// Uses Simpson's 1/3 rule for even intervals, with trapezoidal for odd.
///
/// For constant spacing (x=None), uses efficient tensor operations.
/// For variable spacing, uses element-wise computation.
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

    // For variable spacing, fall back to element-wise computation
    // (tensor ops not efficient for non-uniform weights)
    if let Some(x_val) = x {
        return simpson_variable_spacing(client, y, x_val, n);
    }

    // Constant spacing - use efficient tensor operations
    simpson_constant_spacing(client, y, dx, n)
}

/// Simpson's rule with constant spacing using tensor operations.
fn simpson_constant_spacing<R, C>(
    client: &C,
    y: &Tensor<R>,
    dx: f64,
    n: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();

    // For 2 points, just use trapezoidal
    if n == 2 {
        let sum_y = client.sum(y, &[y_shape.len() - 1], false)?;
        return client.mul_scalar(&sum_y, 0.5 * dx);
    }

    // Number of intervals
    let intervals = n - 1;

    // Create Simpson's weights: [1, 4, 2, 4, 2, ..., 4, 1] for even intervals
    // or [1, 4, 2, 4, 2, ..., 4, 2, 1] plus separate trapezoidal for odd
    let (weights, use_trap) = create_simpson_weights(n, intervals);

    // Create weight tensor
    let weight_shape = if y_shape.len() == 1 {
        vec![n]
    } else {
        // For batched case, broadcast weights
        vec![1, n]
    };
    let weight_tensor = Tensor::<R>::from_slice(&weights, &weight_shape, client.device());

    // Multiply y by weights: y * weights
    let weighted = client.mul(y, &weight_tensor)?;

    // Sum along last dimension
    let sum_axis = y_shape.len() - 1;
    let sum = client.sum(&weighted, &[sum_axis], false)?;

    // Multiply by dx/3
    // Note: For odd intervals, the trapezoidal correction is already baked into the weights
    let result = client.mul_scalar(&sum, dx / 3.0)?;

    let _ = use_trap; // Correction already in weights

    Ok(result)
}

/// Create Simpson's weights for n points.
/// Returns (weights, needs_trapezoidal_correction)
fn create_simpson_weights(n: usize, intervals: usize) -> (Vec<f64>, bool) {
    let mut weights = vec![0.0; n];

    if intervals.is_multiple_of(2) {
        // Even intervals: pure Simpson's 1/3 rule
        // Pattern: 1, 4, 2, 4, 2, ..., 4, 1
        weights[0] = 1.0;
        weights[n - 1] = 1.0;
        for (i, w) in weights.iter_mut().enumerate().skip(1).take(n - 2) {
            *w = if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        (weights, false)
    } else {
        // Odd intervals: Simpson's for n-1 points, then trapezoidal for last
        // For first n-1 points (even intervals): 1, 4, 2, ..., 4, 1
        // Then for last interval: trapezoidal adds 0.5*(y[n-2] + y[n-1])
        //
        // Combined weights:
        // weights[0..n-2] follow Simpson pattern
        // weights[n-2] = Simpson's 1 + trap's 0.5*3/dx... wait, this is tricky
        //
        // Let's compute separately:
        // Simpson's contributes: dx/3 * sum(simpson_weights * y[0..n-1])
        // Trap contributes: dx/2 * (y[n-2] + y[n-1])
        //
        // To combine with single weight vector:
        // Total = dx/3 * (w[0]*y[0] + ... + w[n-1]*y[n-1])
        //
        // For Simpson part (first n-1 points with n-2 intervals):
        // w[0] = 1, w[1] = 4, w[2] = 2, ..., w[n-3] = 4, w[n-2] = 1
        //
        // Trap part: dx/2*(y[n-2] + y[n-1]) = dx/3 * (1.5*y[n-2] + 1.5*y[n-1])
        //
        // So combined:
        // w[n-2] = 1 (Simpson) + 1.5 (trap) = 2.5
        // w[n-1] = 1.5 (trap only)

        // First apply Simpson weights for n-1 points
        weights[0] = 1.0;
        for (i, w) in weights.iter_mut().enumerate().skip(1).take(n - 3) {
            *w = if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        // Adjust last two points for combined Simpson + trapezoidal
        weights[n - 2] = if (n - 2).is_multiple_of(2) { 1.0 } else { 4.0 }; // Simpson contribution
        weights[n - 2] += 1.5; // Trapezoidal contribution
        weights[n - 1] = 1.5; // Trapezoidal contribution only

        (weights, true)
    }
}

/// Simpson's rule with variable spacing.
///
/// For variable spacing, we need element-wise computation because
/// each interval has a different width.
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

    // For variable spacing, the weights depend on the x values
    // This is inherently sequential, so we extract to CPU
    // This is acceptable as variable-spacing quadrature is rare in HPC workloads

    // Extract data (one-time transfer for the quadrature operation)
    let y_data: Vec<f64> = y.to_vec();
    let x_data: Vec<f64> = x.to_vec();

    // For 1D case
    if y_shape.len() == 1 {
        let integral = simpson_1d_variable(&y_data, &x_data);
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row_offset = b * n;
        let row = &y_data[row_offset..row_offset + n];
        let integral = simpson_1d_variable(row, &x_data);
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Simpson's rule for 1D data with variable spacing.
fn simpson_1d_variable(y: &[f64], x: &[f64]) -> f64 {
    let n = y.len();

    if n < 2 {
        return 0.0;
    }

    if n == 2 {
        let h = x[1] - x[0];
        return 0.5 * h * (y[0] + y[1]);
    }

    let intervals = n - 1;

    if intervals.is_multiple_of(2) {
        // Even intervals - pure Simpson's 1/3 rule
        let mut integral = 0.0;
        for i in (0..intervals).step_by(2) {
            let h = (x[i + 2] - x[i]) / 2.0;
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }
        integral
    } else {
        // Odd intervals - Simpson's for most, trapezoidal for last
        let mut integral = 0.0;

        // Simpson's for even part
        for i in (0..intervals - 1).step_by(2) {
            let h = (x[i + 2] - x[i]) / 2.0;
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }

        // Trapezoidal for last interval
        let h = x[n - 1] - x[n - 2];
        integral += 0.5 * h * (y[n - 2] + y[n - 1]);

        integral
    }
}
