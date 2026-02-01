//! Tensor-based quadrature implementations.

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trapezoidal rule integration using tensors.
///
/// Computes ∫y dx using the composite trapezoidal rule.
/// For 1D tensors, returns a 0-D tensor with the integral.
/// For 2D tensors, integrates each row and returns a 1D tensor.
pub fn trapezoid_impl<R, C>(client: &C, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
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

    // Get data as vectors for computation
    let y_data: Vec<f64> = y.to_vec();
    let x_data: Vec<f64> = x.to_vec();

    // For 1D case
    if y_shape.len() == 1 {
        let mut integral = 0.0;
        for i in 0..n - 1 {
            let dx = x_data[i + 1] - x_data[i];
            integral += 0.5 * dx * (y_data[i] + y_data[i + 1]);
        }
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration - integrate each row)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let mut integral = 0.0;
        let row_offset = b * n;
        for i in 0..n - 1 {
            let dx = x_data[i + 1] - x_data[i];
            integral += 0.5 * dx * (y_data[row_offset + i] + y_data[row_offset + i + 1]);
        }
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Trapezoidal rule with uniform spacing.
pub fn trapezoid_uniform_impl<R, C>(client: &C, y: &Tensor<R>, dx: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
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

    let y_data: Vec<f64> = y.to_vec();

    // For 1D case
    if y_shape.len() == 1 {
        let mut integral = 0.5 * dx * (y_data[0] + y_data[n - 1]);
        for &val in &y_data[1..n - 1] {
            integral += dx * val;
        }
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row_offset = b * n;
        let mut integral = 0.5 * dx * (y_data[row_offset] + y_data[row_offset + n - 1]);
        for i in 1..n - 1 {
            integral += dx * y_data[row_offset + i];
        }
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Cumulative trapezoidal integration.
///
/// Returns a tensor of the same shape as y with cumulative integrals.
pub fn cumulative_trapezoid_impl<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: Option<&Tensor<R>>,
    dx: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
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

    let y_data: Vec<f64> = y.to_vec();
    let x_data: Option<Vec<f64>> = x.map(|t| t.to_vec());

    // For 1D case
    if y_shape.len() == 1 {
        let mut result = vec![0.0; n - 1];
        let mut cumsum = 0.0;

        for i in 0..n - 1 {
            let step_dx = if let Some(ref xd) = x_data {
                xd[i + 1] - xd[i]
            } else {
                dx
            };
            cumsum += 0.5 * step_dx * (y_data[i] + y_data[i + 1]);
            result[i] = cumsum;
        }

        return Ok(Tensor::<R>::from_slice(&result, &[n - 1], client.device()));
    }

    // For 2D case
    let batch_size = y_shape[0];
    let out_n = n - 1;
    let mut result = vec![0.0; batch_size * out_n];

    for b in 0..batch_size {
        let row_offset = b * n;
        let out_offset = b * out_n;
        let mut cumsum = 0.0;

        for i in 0..n - 1 {
            let step_dx = if let Some(ref xd) = x_data {
                xd[i + 1] - xd[i]
            } else {
                dx
            };
            cumsum += 0.5 * step_dx * (y_data[row_offset + i] + y_data[row_offset + i + 1]);
            result[out_offset + i] = cumsum;
        }
    }

    Ok(Tensor::<R>::from_slice(
        &result,
        &[batch_size, out_n],
        client.device(),
    ))
}

/// Simpson's rule integration.
///
/// Uses Simpson's 1/3 rule for even intervals, with trapezoidal for odd.
pub fn simpson_impl<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: Option<&Tensor<R>>,
    dx: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
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

    let y_data: Vec<f64> = y.to_vec();
    let x_data: Option<Vec<f64>> = x.map(|t| t.to_vec());

    // For 1D case
    if y_shape.len() == 1 {
        let integral = simpson_1d(&y_data, x_data.as_deref(), dx);
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row_offset = b * n;
        let row = &y_data[row_offset..row_offset + n];
        let integral = simpson_1d(row, x_data.as_deref(), dx);
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Simpson's rule for 1D data.
fn simpson_1d(y: &[f64], x: Option<&[f64]>, dx: f64) -> f64 {
    let n = y.len();

    if n < 2 {
        return 0.0;
    }

    if n == 2 {
        // Just use trapezoidal for 2 points
        let h = x.map_or(dx, |xd| xd[1] - xd[0]);
        return 0.5 * h * (y[0] + y[1]);
    }

    // Number of intervals
    let intervals = n - 1;

    if intervals.is_multiple_of(2) {
        // Even number of intervals - pure Simpson's 1/3 rule
        let mut integral = 0.0;
        for i in (0..intervals).step_by(2) {
            let h = x.map_or(dx, |xd| (xd[i + 2] - xd[i]) / 2.0);
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }
        integral
    } else {
        // Odd number of intervals - use Simpson's for most, trapezoidal for last
        let mut integral = 0.0;

        // Simpson's for even part
        for i in (0..intervals - 1).step_by(2) {
            let h = x.map_or(dx, |xd| (xd[i + 2] - xd[i]) / 2.0);
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }

        // Trapezoidal for last interval
        let h = x.map_or(dx, |xd| xd[n - 1] - xd[n - 2]);
        integral += 0.5 * h * (y[n - 2] + y[n - 1]);

        integral
    }
}

/// Fixed-order Gaussian quadrature.
///
/// Integrates a function from a to b using n-point Gauss-Legendre quadrature.
pub fn fixed_quad_impl<R, C, F>(client: &C, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "fixed_quad: n must be at least 1".to_string(),
        });
    }

    // Get Gauss-Legendre nodes and weights for [-1, 1]
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform nodes from [-1, 1] to [a, b]
    let half_width = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let transformed_nodes: Vec<f64> = nodes.iter().map(|&x| center + half_width * x).collect();

    // Evaluate function at all nodes
    let x_tensor = Tensor::<R>::from_slice(&transformed_nodes, &[n], client.device());
    let f_values = f(&x_tensor)?;
    let f_data: Vec<f64> = f_values.to_vec();

    // Compute weighted sum
    let mut integral = 0.0;
    for i in 0..n {
        integral += weights[i] * f_data[i];
    }
    integral *= half_width;

    Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()))
}

/// Compute Gauss-Legendre nodes and weights.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let m = n.div_ceil(2);

    for i in 0..m {
        // Initial guess using Chebyshev approximation
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();

        // Newton iteration to find root of Legendre polynomial
        loop {
            let (p, dp) = legendre_p_and_dp(n, z);
            let z_new = z - p / dp;

            if (z_new - z).abs() < 1e-15 {
                z = z_new;
                break;
            }
            z = z_new;
        }

        let (_, dp) = legendre_p_and_dp(n, z);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);

        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

/// Evaluate Legendre polynomial P_n(x) and its derivative.
fn legendre_p_and_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;

    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    // Derivative: P'_n(x) = n * (x * P_n - P_{n-1}) / (x^2 - 1)
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);

    (p_curr, dp)
}
