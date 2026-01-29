//! Shared utility functions for optimization algorithms.
//!
//! This module provides common numerical operations used across multiple
//! optimization algorithms to avoid code duplication.

#![allow(dead_code)] // Utility functions may not all be used yet

/// Numerical threshold for detecting singular matrices.
pub const SINGULAR_THRESHOLD: f64 = 1e-14;

/// Threshold for treating values as effectively zero.
pub const ZERO_THRESHOLD: f64 = 1e-10;

/// Default epsilon for finite difference approximations.
pub const DEFAULT_FINITE_DIFF_EPS: f64 = 1e-8;

/// Compute the L2 (Euclidean) norm of a vector.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// The L2 norm: sqrt(sum(v_i^2))
#[inline]
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the squared L2 norm of a vector.
///
/// More efficient than `norm()` when only comparing magnitudes.
#[inline]
pub fn norm_squared(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the infinity norm (max absolute value) of a vector.
#[inline]
pub fn norm_inf(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
///
/// # Arguments
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (length n)
///
/// # Returns
/// * `Some(x)` - Solution vector if system is non-singular
/// * `None` - If matrix is singular or dimensions don't match
#[allow(clippy::needless_range_loop)]
pub fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    // Create augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot (row with maximum absolute value in current column)
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < SINGULAR_THRESHOLD {
            return None; // Singular matrix
        }

        // Swap rows if necessary
        aug.swap(col, max_row);

        // Eliminate entries below pivot
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Compute finite difference Jacobian approximation for a vector-valued function.
///
/// Uses forward differences: J[i,j] = (F_i(x + eps*e_j) - F_i(x)) / eps
///
/// # Arguments
/// * `f` - Function F: R^n -> R^m
/// * `x` - Point at which to evaluate Jacobian
/// * `fx` - Function value at x (to avoid recomputation)
/// * `eps` - Step size for finite differences
///
/// # Returns
/// Jacobian matrix (m x n) where J[i][j] = ∂F_i/∂x_j
pub fn finite_difference_jacobian<F>(f: &F, x: &[f64], fx: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = fx.len();
    let mut jacobian = vec![vec![0.0; n]; m];
    let mut x_pert = x.to_vec();

    for j in 0..n {
        let x_orig = x_pert[j];
        x_pert[j] = x_orig + eps;
        let fx_pert = f(&x_pert);
        x_pert[j] = x_orig;

        for i in 0..m {
            jacobian[i][j] = (fx_pert[i] - fx[i]) / eps;
        }
    }

    jacobian
}

/// Compute finite difference gradient approximation for a scalar-valued function.
///
/// Uses central differences: g[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
///
/// # Arguments
/// * `f` - Function f: R^n -> R
/// * `x` - Point at which to evaluate gradient
/// * `eps` - Step size for finite differences
///
/// # Returns
/// Gradient vector (length n) where g[i] = ∂f/∂x_i
pub fn finite_difference_gradient<F>(f: &F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_pert = x.to_vec();

    for i in 0..n {
        let x_orig = x_pert[i];

        // Central difference
        x_pert[i] = x_orig + eps;
        let f_plus = f(&x_pert);

        x_pert[i] = x_orig - eps;
        let f_minus = f(&x_pert);

        x_pert[i] = x_orig;

        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }

    grad
}

/// Compute finite difference gradient using forward differences.
///
/// Less accurate than central differences but requires fewer function evaluations.
///
/// # Arguments
/// * `f` - Function f: R^n -> R
/// * `x` - Point at which to evaluate gradient
/// * `fx` - Function value at x (to avoid recomputation)
/// * `eps` - Step size for finite differences
///
/// # Returns
/// Gradient vector (length n) where g[i] = ∂f/∂x_i
pub fn finite_difference_gradient_forward<F>(f: &F, x: &[f64], fx: f64, eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_pert = x.to_vec();

    for i in 0..n {
        let x_orig = x_pert[i];
        x_pert[i] = x_orig + eps;
        let f_pert = f(&x_pert);
        x_pert[i] = x_orig;

        grad[i] = (f_pert - fx) / eps;
    }

    grad
}

/// Dot product of two vectors.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector subtraction: a - b
#[inline]
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Vector addition: a + b
#[inline]
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scalar multiplication: alpha * v
#[inline]
pub fn vec_scale(v: &[f64], alpha: f64) -> Vec<f64> {
    v.iter().map(|x| x * alpha).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm() {
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((norm(&[1.0, 1.0, 1.0]) - 3.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(norm(&[]), 0.0);
    }

    #[test]
    fn test_norm_inf() {
        assert_eq!(norm_inf(&[1.0, -5.0, 3.0]), 5.0);
        assert_eq!(norm_inf(&[]), 0.0);
    }

    #[test]
    fn test_solve_linear_system_simple() {
        // 2x + y = 5
        // x + 3y = 5
        // Solution: x = 2, y = 1
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 5.0];
        let x = solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_singular() {
        // Singular matrix (rows are linearly dependent)
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(solve_linear_system(&a, &b).is_none());
    }

    #[test]
    fn test_finite_difference_gradient() {
        // f(x, y) = x^2 + y^2, gradient at (1, 2) is (2, 4)
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = finite_difference_gradient(&f, &[1.0, 2.0], 1e-6);
        assert!((grad[0] - 2.0).abs() < 1e-4);
        assert!((grad[1] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_finite_difference_jacobian() {
        // F(x, y) = (x + y, x - y), Jacobian is [[1, 1], [1, -1]]
        let f = |x: &[f64]| vec![x[0] + x[1], x[0] - x[1]];
        let fx = f(&[1.0, 2.0]);
        let jac = finite_difference_jacobian(&f, &[1.0, 2.0], &fx, 1e-6);
        assert!((jac[0][0] - 1.0).abs() < 1e-4);
        assert!((jac[0][1] - 1.0).abs() < 1e-4);
        assert!((jac[1][0] - 1.0).abs() < 1e-4);
        assert!((jac[1][1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_ops() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert_eq!(vec_add(&a, &b), vec![4.0, 6.0]);
        assert_eq!(vec_sub(&a, &b), vec![-2.0, -2.0]);
        assert_eq!(vec_scale(&a, 2.0), vec![2.0, 4.0]);
    }
}
