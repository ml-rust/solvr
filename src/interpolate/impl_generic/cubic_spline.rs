//! Cubic spline coefficient computation generic implementation.
//!
//! Implements the Thomas algorithm for solving the tridiagonal system
//! that arises in cubic spline interpolation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::cubic_spline::SplineBoundary;
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute cubic spline coefficients using Thomas algorithm.
pub fn cubic_spline_coefficients<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    boundary: &SplineBoundary,
) -> InterpolateResult<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();

    // Get data as vectors for coefficient computation (construction time)
    // This is acceptable here because the tridiagonal solve is inherently sequential
    let x_data: Vec<f64> = x.contiguous().to_vec();
    let y_data: Vec<f64> = y.contiguous().to_vec();

    let n = x_data.len();

    // a coefficients are just the y values
    let a: Vec<f64> = y_data.to_vec();

    // Compute interval widths h_i = x_{i+1} - x_i
    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x_data[i + 1] - x_data[i]);
    }

    // Set up tridiagonal system for c coefficients (second derivatives / 2)
    // The system is: lower[i] * c[i-1] + diag[i] * c[i] + upper[i] * c[i+1] = rhs[i]
    let mut diag = vec![0.0; n];
    let mut upper = vec![0.0; n - 1];
    let mut lower = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];

    // Interior equations (natural cubic spline continuity)
    for i in 1..n - 1 {
        lower[i - 1] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        rhs[i] =
            3.0 * ((y_data[i + 1] - y_data[i]) / h[i] - (y_data[i] - y_data[i - 1]) / h[i - 1]);
    }

    // Apply boundary conditions
    match boundary {
        SplineBoundary::Natural => {
            // c[0] = 0, c[n-1] = 0
            diag[0] = 1.0;
            rhs[0] = 0.0;
            diag[n - 1] = 1.0;
            rhs[n - 1] = 0.0;
        }
        SplineBoundary::Clamped { left, right } => {
            // First derivative specified at endpoints
            // S'(x_0) = left => b[0] = left
            // S'(x_{n-1}) = right => b[n-1] = right
            diag[0] = 2.0 * h[0];
            upper[0] = h[0];
            rhs[0] = 3.0 * ((y_data[1] - y_data[0]) / h[0] - left);

            diag[n - 1] = 2.0 * h[n - 2];
            lower[n - 2] = h[n - 2];
            rhs[n - 1] = 3.0 * (right - (y_data[n - 1] - y_data[n - 2]) / h[n - 2]);
        }
        SplineBoundary::NotAKnot => {
            if n < 4 {
                // Fall back to natural for small n
                diag[0] = 1.0;
                rhs[0] = 0.0;
                diag[n - 1] = 1.0;
                rhs[n - 1] = 0.0;
            } else {
                // Not-a-knot: d[0] = d[1] and d[n-3] = d[n-2]
                // This makes third derivative continuous at x[1] and x[n-2]
                let h0h1 = h[0] * h[0] * h[1];
                let h1h0 = h[1] * h[1] * h[0];
                diag[0] = h1h0;
                upper[0] = -(h0h1 + h1h0);
                rhs[0] = h0h1 * ((y_data[2] - y_data[1]) / h[1] - (y_data[1] - y_data[0]) / h[0]);

                let hn2 = h[n - 2];
                let hn3 = h[n - 3];
                diag[n - 1] = hn3 * hn3 * hn2;
                lower[n - 2] = -(hn2 * hn2 * hn3 + hn3 * hn3 * hn2);
                rhs[n - 1] = hn2
                    * hn2
                    * hn3
                    * ((y_data[n - 1] - y_data[n - 2]) / hn2
                        - (y_data[n - 2] - y_data[n - 3]) / hn3);
            }
        }
    }

    // Solve tridiagonal system using Thomas algorithm
    let c = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

    // Compute b and d coefficients from c
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];

    for i in 0..n - 1 {
        b[i] = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
    }

    // Convert to tensors
    let a_tensor = Tensor::from_slice(&a, &[n], device);
    let b_tensor = Tensor::from_slice(&b, &[n - 1], device);
    let c_tensor = Tensor::from_slice(&c, &[n], device);
    let d_tensor = Tensor::from_slice(&d, &[n - 1], device);

    Ok((a_tensor, b_tensor, c_tensor, d_tensor))
}

/// Solve tridiagonal system using Thomas algorithm.
fn solve_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> InterpolateResult<Vec<f64>> {
    let n = diag.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i.saturating_sub(1)] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            return Err(InterpolateError::NumericalError {
                message: "Singular tridiagonal system in spline computation".to_string(),
            });
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (rhs[i] - lower[i.saturating_sub(1)] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_tridiagonal_solver() {
        // Simple 3x3 system
        let lower = vec![1.0, 1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let upper = vec![1.0, 1.0];
        let rhs = vec![1.0, 2.0, 1.0];

        let x = solve_tridiagonal(&lower, &diag, &upper, &rhs).unwrap();

        // Verify solution satisfies Ax = b
        assert!((diag[0] * x[0] + upper[0] * x[1] - rhs[0]).abs() < 1e-10);
        assert!((lower[0] * x[0] + diag[1] * x[1] + upper[1] * x[2] - rhs[1]).abs() < 1e-10);
        assert!((lower[1] * x[1] + diag[2] * x[2] - rhs[2]).abs() < 1e-10);
    }
}
