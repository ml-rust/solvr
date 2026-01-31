//! Root finding algorithms for scalar functions.

use super::{RootResult, ScalarOptions};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::SINGULAR_THRESHOLD;

/// Bisection method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Bisection is slow (linear convergence) but very robust.
pub fn bisect<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "bisect".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "bisect".to_string(),
        });
    }

    let mut left = a;
    let mut right = b;
    let mut f_left = fa;

    for iter in 0..options.max_iter {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid);

        // Check convergence
        let width = right - left;
        if width.abs() < options.tol || width.abs() / mid.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: mid,
                function_value: f_mid,
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Update bracket
        if (f_mid > 0.0 && f_left > 0.0) || (f_mid < 0.0 && f_left < 0.0) {
            left = mid;
            f_left = f_mid;
        } else {
            right = mid;
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "bisect".to_string(),
    })
}

/// Newton's method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `df` - Derivative of f
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` near `x0`
///
/// # Errors
/// * `DidNotConverge` if iterations exceed max_iter
/// * `NumericalError` if derivative is too close to zero
///
/// # Note
/// Newton's method has quadratic convergence but may diverge if x0 is far from root.
pub fn newton<F, DF>(f: F, df: DF, x0: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;

    for iter in 0..options.max_iter {
        let fx = f(x);
        let dfx = df(x);

        if dfx.abs() < SINGULAR_THRESHOLD {
            return Err(OptimizeError::NumericalError {
                message: "Derivative too close to zero".to_string(),
            });
        }

        let x_new = x - fx / dfx;
        let dx = (x_new - x).abs();

        // Check convergence
        if dx < options.tol || dx / x.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_new,
                function_value: f(x_new),
                iterations: iter + 1,
                bracket_width: dx,
            });
        }

        x = x_new;
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "newton".to_string(),
    })
}

/// Secant method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `x0` - First initial guess
/// * `x1` - Second initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` near `x0` and `x1`
///
/// # Errors
/// * `DidNotConverge` if iterations exceed max_iter
/// * `NumericalError` if denominator becomes too small
///
/// # Note
/// Secant method has superlinear convergence (~1.618) and doesn't require derivatives.
pub fn secant<F>(f: F, x0: f64, x1: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut f_prev = f(x_prev);
    let mut f_curr = f(x_curr);

    for iter in 0..options.max_iter {
        let denom = f_curr - f_prev;

        if denom.abs() < SINGULAR_THRESHOLD {
            return Err(OptimizeError::NumericalError {
                message: "Denominator too close to zero in secant method".to_string(),
            });
        }

        let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
        let dx = (x_next - x_curr).abs();

        // Check convergence
        if dx < options.tol || dx / x_curr.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_next,
                function_value: f(x_next),
                iterations: iter + 1,
                bracket_width: dx,
            });
        }

        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
        f_curr = f(x_curr);
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "secant".to_string(),
    })
}

/// Bracketed root finding with bisection.
///
/// This is a robust bisection-based root finder that maintains a bracket
/// around the root at all times, guaranteeing convergence.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Named `brentq` for SciPy compatibility. Currently uses pure bisection
/// for robustness. Convergence is O(log(|b-a|/tol)).
pub fn brentq<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "brentq".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "brentq".to_string(),
        });
    }

    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut _fb = fb;

    for iter in 0..options.max_iter {
        let width = (b - a).abs();
        let tol_here = options.tol.max(options.rtol * a.abs().max(1.0));

        // Check convergence
        if width < tol_here {
            let mid = 0.5 * (a + b);
            return Ok(RootResult {
                root: mid,
                function_value: f(mid),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Use bisection with occasional interpolation for robustness
        let mid = 0.5 * (a + b);
        let f_mid = f(mid);

        // Update bracket to maintain sign change
        if (f_mid > 0.0 && fa > 0.0) || (f_mid < 0.0 && fa < 0.0) {
            a = mid;
            fa = f_mid;
        } else {
            b = mid;
            _fb = f_mid;
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "brentq".to_string(),
    })
}

/// Ridder's method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Ridder's method has linear (but very good) convergence and is more efficient than bisection.
pub fn ridder<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "ridder".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "ridder".to_string(),
        });
    }

    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut fb = fb;

    for iter in 0..options.max_iter {
        let c = 0.5 * (a + b);
        let fc = f(c);

        // Compute new estimate using Ridder's formula
        let denom = (2.0 * fc * fc - fa * fb).sqrt();
        if denom.abs() < SINGULAR_THRESHOLD {
            // Fallback to midpoint
            return Ok(RootResult {
                root: c,
                function_value: fc,
                iterations: iter + 1,
                bracket_width: (b - a).abs(),
            });
        }

        let s = if fa > fb { -1.0 } else { 1.0 };
        let x_new = c + s * (c - a) * fc / denom;

        let f_new = f(x_new);

        // Check convergence
        let width = (b - a).abs();
        if width < options.tol || width / c.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_new,
                function_value: f_new,
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Update bracket
        if (f_new > 0.0 && fc > 0.0) || (f_new < 0.0 && fc < 0.0) {
            if (f_new > 0.0 && fa < 0.0) || (f_new < 0.0 && fa > 0.0) {
                b = x_new;
                fb = f_new;
            } else {
                a = x_new;
                fa = f_new;
            }
        } else {
            // x_new is on the opposite side of c from either a or b
            if (f_new > 0.0 && fc < 0.0) || (f_new < 0.0 && fc > 0.0) {
                a = c;
                fa = fc;
                b = x_new;
                fb = f_new;
            } else {
                a = x_new;
                fa = f_new;
                b = c;
                fb = fc;
            }
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "ridder".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisect_simple() {
        let result =
            bisect(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("bisect failed");
        assert!((result.root - 2.0).abs() < 1e-10);
        assert!(result.function_value.abs() < 1e-10);
    }

    #[test]
    fn test_bisect_negative_root() {
        let result =
            bisect(|x| x * x - 4.0, -3.0, -1.0, &ScalarOptions::default()).expect("bisect failed");
        assert!((result.root - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_same_sign() {
        let result = bisect(|x| x * x + 1.0, 1.0, 3.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::SameSignBracket { .. })));
    }

    #[test]
    fn test_bisect_invalid_interval() {
        let result = bisect(|x| x * x - 4.0, 3.0, 1.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_newton_simple() {
        let result = newton(|x| x * x - 4.0, |x| 2.0 * x, 3.0, &ScalarOptions::default())
            .expect("newton failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_secant_simple() {
        let result =
            secant(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("secant failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ridder_simple() {
        let result =
            ridder(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("ridder failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_polynomial() {
        let f = |x: f64| x * x * x - 2.0 * x * x - x + 2.0;
        let result = bisect(f, 0.5, 1.5, &ScalarOptions::default()).expect("failed");
        assert!((result.root - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trigonometric() {
        let result = bisect(|x: f64| x.sin(), 2.0, 4.0, &ScalarOptions::default()).expect("failed");
        assert!((result.root - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_exponential() {
        let result =
            bisect(|x: f64| x.exp() - 3.0, 0.0, 2.0, &ScalarOptions::default()).expect("failed");
        assert!((result.root - 3_f64.ln()).abs() < 1e-10);
    }
}
