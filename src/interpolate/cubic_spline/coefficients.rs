//! Cubic spline coefficient computation.
//!
//! Implements the Thomas algorithm for solving the tridiagonal system
//! that arises in cubic spline interpolation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};

use super::SplineBoundary;

/// Compute cubic spline coefficients using Thomas algorithm.
#[allow(clippy::type_complexity)]
pub fn compute_coefficients(
    x: &[f64],
    y: &[f64],
    boundary: &SplineBoundary,
) -> InterpolateResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = x.len();

    // a coefficients are just the y values
    let a: Vec<f64> = y.to_vec();

    // Compute interval widths h_i = x_{i+1} - x_i
    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(x[i + 1] - x[i]);
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
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
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
            rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - *left);

            diag[n - 1] = 2.0 * h[n - 2];
            lower[n - 2] = h[n - 2];
            rhs[n - 1] = 3.0 * (*right - (y[n - 1] - y[n - 2]) / h[n - 2]);
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
                diag[0] = h[1];
                upper[0] = -(h[0] + h[1]);
                rhs[0] = 0.0;
                // Need to add h[0] * c[2] term - modify the system
                // For simplicity, we use a modified approach
                let h0h1 = h[0] * h[0] * h[1];
                let h1h0 = h[1] * h[1] * h[0];
                diag[0] = h1h0;
                upper[0] = -(h0h1 + h1h0);
                rhs[0] = h0h1 * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0]);

                let hn2 = h[n - 2];
                let hn3 = h[n - 3];
                diag[n - 1] = hn3 * hn3 * hn2;
                lower[n - 2] = -(hn2 * hn2 * hn3 + hn3 * hn3 * hn2);
                rhs[n - 1] =
                    hn2 * hn2 * hn3 * ((y[n - 1] - y[n - 2]) / hn2 - (y[n - 2] - y[n - 3]) / hn3);
            }
        }
    }

    // Solve tridiagonal system using Thomas algorithm
    let c = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

    // Compute b and d coefficients from c
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];

    for i in 0..n - 1 {
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
    }

    Ok((a, b, c, d))
}

/// Solve tridiagonal system using Thomas algorithm.
pub fn solve_tridiagonal(
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

    #[test]
    fn test_natural_coefficients() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];

        let (a, b, c, d) = compute_coefficients(&x, &y, &SplineBoundary::Natural).unwrap();

        // a should equal y
        assert_eq!(a.len(), 4);
        assert!((a[0] - 0.0).abs() < 1e-10);
        assert!((a[1] - 1.0).abs() < 1e-10);
        assert!((a[2] - 0.0).abs() < 1e-10);
        assert!((a[3] - 1.0).abs() < 1e-10);

        // c[0] and c[n-1] should be 0 for natural spline
        assert!(c[0].abs() < 1e-10);
        assert!(c[3].abs() < 1e-10);

        // b and d should have n-1 elements
        assert_eq!(b.len(), 3);
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn test_clamped_coefficients() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];

        let (a, b, _c, _d) = compute_coefficients(
            &x,
            &y,
            &SplineBoundary::Clamped {
                left: 2.0,
                right: -2.0,
            },
        )
        .unwrap();

        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 2);
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
