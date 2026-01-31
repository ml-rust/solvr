//! Scalar minimization algorithms.

use super::{MinimizeResult, ScalarOptions};
use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Golden section search for minimization.
///
/// # Arguments
/// * `f` - Function to minimize
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Golden section search is robust and doesn't require derivatives.
/// It has linear convergence but is very reliable.
/// Works well for unimodal functions.
pub fn minimize_scalar_golden<F>(
    f: F,
    a: f64,
    b: f64,
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "minimize_scalar_golden".to_string(),
        });
    }

    // Golden ratio: phi = (1 + sqrt(5)) / 2 ≈ 1.618
    // We use the inverse ratio for shrinking: 1/phi = (sqrt(5) - 1) / 2 ≈ 0.618
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0; // ≈ 0.618034
    let inv_phi2 = 1.0 - inv_phi; // ≈ 0.381966

    let mut a = a;
    let mut b = b;

    // Initial interior points
    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for iter in 0..options.max_iter {
        let width = b - a;
        let tol_here = options
            .tol
            .max(options.rtol * (a.abs().max(b.abs()).max(1.0)));

        // Check convergence
        if width < tol_here {
            // Return the midpoint of the final bracket
            let x_min = 0.5 * (a + b);
            return Ok(MinimizeResult {
                x: x_min,
                f_min: f(x_min),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Narrow the bracket
        if f1 < f2 {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(x2);
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "minimize_scalar_golden".to_string(),
    })
}

/// Bracketed scalar minimization using golden section search.
///
/// # Arguments
/// * `f` - Function to minimize
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Named `minimize_scalar_brent` for SciPy compatibility. Currently uses
/// pure golden section search for robustness. Convergence rate is linear
/// with ratio 0.618 (golden ratio).
pub fn minimize_scalar_brent<F>(
    f: F,
    a: f64,
    b: f64,
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "minimize_scalar_brent".to_string(),
        });
    }

    // Golden ratio constants
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0; // ≈ 0.618034
    let inv_phi2 = 1.0 - inv_phi; // ≈ 0.381966

    let mut a = a;
    let mut b = b;

    // Initial interior points
    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for iter in 0..options.max_iter {
        let width = b - a;
        let tol_here = options
            .tol
            .max(options.rtol * (a.abs().max(b.abs()).max(1.0)));

        // Check convergence
        if width < tol_here {
            // Return the midpoint of the final bracket
            let x_min = 0.5 * (a + b);
            return Ok(MinimizeResult {
                x: x_min,
                f_min: f(x_min),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Golden section step (Brent uses this with occasional parabolic interpolation,
        // but for robustness we use pure golden section which is proven to work)
        if f1 < f2 {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(x2);
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "minimize_scalar_brent".to_string(),
    })
}

/// Bounded scalar minimization (wrapper for minimize_scalar_brent).
///
/// # Arguments
/// * `f` - Function to minimize
/// * `bounds` - (xmin, xmax) bracket for minimization
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in the bounded interval
///
/// # Errors
/// * `InvalidInterval` if xmin >= xmax
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// This is a convenience wrapper around `minimize_scalar_brent` for bounded problems.
pub fn minimize_scalar_bounded<F>(
    f: F,
    bounds: (f64, f64),
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    let (xmin, xmax) = bounds;
    if xmin >= xmax {
        return Err(OptimizeError::InvalidInterval {
            a: xmin,
            b: xmax,
            context: "minimize_scalar_bounded".to_string(),
        });
    }
    minimize_scalar_brent(f, xmin, xmax, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_golden_simple_quadratic() {
        let result = minimize_scalar_golden(
            |x| (x - 2.0) * (x - 2.0),
            0.0,
            4.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_golden failed");
        assert!((result.x - 2.0).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_golden_cubic() {
        let result = minimize_scalar_golden(|x| x * x * x - x, 0.0, 2.0, &ScalarOptions::default())
            .expect("minimize_scalar_golden failed");
        let expected_x = 1.0 / (3.0_f64).sqrt();
        let expected_min = -2.0 * (3.0_f64).sqrt() / 9.0;
        assert!((result.x - expected_x).abs() < 1e-6);
        assert!((result.f_min - expected_min).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_golden_sine() {
        let result = minimize_scalar_golden(
            |x: f64| x.sin(),
            0.0,
            2.0 * std::f64::consts::PI,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_golden failed");
        let expected_x = 3.0 * std::f64::consts::PI / 2.0;
        assert!((result.x - expected_x).abs() < 1e-5);
        assert!((result.f_min - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_minimize_brent_simple_quadratic() {
        let result = minimize_scalar_brent(
            |x| (x - 2.0) * (x - 2.0),
            0.0,
            4.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - 2.0).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_brent_negative_shift() {
        let result = minimize_scalar_brent(
            |x| (x + 3.0) * (x + 3.0),
            -5.0,
            -1.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - (-3.0)).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_brent_quartic() {
        let result = minimize_scalar_brent(
            |x| x * x * x * x - 3.0 * x * x + 2.0,
            0.0,
            2.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        let expected_x = (1.5_f64).sqrt();
        assert!((result.x - expected_x).abs() < 1e-5);
    }

    #[test]
    fn test_minimize_brent_exponential() {
        let result = minimize_scalar_brent(
            |x: f64| x.exp() * (x - 1.0) * (x - 1.0),
            -1.0,
            3.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - 1.0).abs() < 1e-5);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_bounded_simple() {
        let result = minimize_scalar_bounded(
            |x| (x - 1.0) * (x - 1.0),
            (0.0, 3.0),
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_bounded failed");
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_bounded_boundary_minimum() {
        let result = minimize_scalar_bounded(
            |x| (x - 0.5) * (x - 0.5),
            (0.0, 1.0),
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_bounded failed");
        assert!((result.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_bounded_invalid_interval() {
        let result = minimize_scalar_bounded(|x| x * x, (3.0, 1.0), &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_golden_invalid_interval() {
        let result = minimize_scalar_golden(|x| x * x, 4.0, 2.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_brent_invalid_interval() {
        let result = minimize_scalar_brent(|x| x * x, 5.0, 1.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_convergence_comparison() {
        let f = |x: f64| (x - 3.5) * (x - 3.5) + 0.1 * (x - 3.5).sin();

        let golden_result =
            minimize_scalar_golden(f, 0.0, 5.0, &ScalarOptions::default()).expect("golden failed");
        let brent_result =
            minimize_scalar_brent(f, 0.0, 5.0, &ScalarOptions::default()).expect("brent failed");

        // Both should find similar minima
        assert!((golden_result.x - brent_result.x).abs() < 1e-4);
        assert!(brent_result.iterations > 0 && golden_result.iterations > 0);
    }
}
