//! Shared Hermite cubic interpolation primitives.
//!
//! This module provides the common building blocks for Hermite-based interpolators
//! (PCHIP, Akima, etc.). It eliminates code duplication by centralizing:
//! - Hermite basis function evaluation
//! - Hermite derivative evaluation
//! - Binary search interval finding
//! - Input validation
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
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validated input data from 1D interpolation tensors.
pub struct ValidatedData {
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub n: usize,
    pub x_min: f64,
    pub x_max: f64,
}

/// Data required for Hermite interpolation evaluation.
///
/// Bundles all parameters needed by `evaluate_hermite` and `derivative_hermite`
/// to avoid excessive function arguments.
pub struct HermiteData<'a> {
    pub x_data: &'a [f64],
    pub y_data: &'a [f64],
    pub slopes: &'a [f64],
    pub n: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub context: &'a str,
}

/// Data required for Hermite interpolation at a point.
pub struct HermitePoint {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
    pub d0: f64,
    pub d1: f64,
}

/// Evaluate Hermite cubic at a point within an interval.
///
/// # Arguments
/// * `point` - The interval data (endpoints, values, slopes)
/// * `xi` - The x coordinate to evaluate at (must be in [x0, x1])
#[inline]
pub fn hermite_eval(point: &HermitePoint, xi: f64) -> f64 {
    let h = point.x1 - point.x0;
    let t = (xi - point.x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    h00 * point.y0 + h10 * h * point.d0 + h01 * point.y1 + h11 * h * point.d1
}

/// Evaluate the derivative of Hermite cubic at a point.
///
/// # Arguments
/// * `point` - The interval data (endpoints, values, slopes)
/// * `xi` - The x coordinate to evaluate at (must be in [x0, x1])
#[inline]
pub fn hermite_derivative(point: &HermitePoint, xi: f64) -> f64 {
    let h = point.x1 - point.x0;
    let t = (xi - point.x0) / h;
    let t2 = t * t;

    // Derivatives of Hermite basis functions (with chain rule factor 1/h)
    let dh00 = (6.0 * t2 - 6.0 * t) / h;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = (-6.0 * t2 + 6.0 * t) / h;
    let dh11 = 3.0 * t2 - 2.0 * t;

    dh00 * point.y0 + dh10 * point.d0 + dh01 * point.y1 + dh11 * point.d1
}

/// Find the interval index for a given x value using binary search.
///
/// Returns the index `i` such that `x_data[i] <= xi < x_data[i+1]`,
/// or the last valid interval index if xi equals the maximum.
#[inline]
pub fn find_interval(x_data: &[f64], n: usize, xi: f64) -> usize {
    let mut lo = 0;
    let mut hi = n - 1;

    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if x_data[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
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
) -> InterpolateResult<ValidatedData> {
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

    // Get data as vectors
    let x_data: Vec<f64> = x.to_vec();
    let y_data: Vec<f64> = y.to_vec();

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
        x_data,
        y_data,
        n,
        x_min,
        x_max,
    })
}

/// Evaluate a Hermite interpolant at multiple points.
///
/// This is a generic evaluation function used by PCHIP, Akima, etc.
pub fn evaluate_hermite<R: Runtime, C: RuntimeClient<R>>(
    _client: &C,
    x_new: &Tensor<R>,
    data: &HermiteData<'_>,
) -> InterpolateResult<Tensor<R>> {
    let x_new_shape = x_new.shape();
    if x_new_shape.len() != 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "x_new".to_string(),
            message: "x_new must be a 1D tensor".to_string(),
        });
    }

    let x_new_data: Vec<f64> = x_new.to_vec();
    let mut y_new_data = Vec::with_capacity(x_new_data.len());

    for &xi in &x_new_data {
        if xi < data.x_min || xi > data.x_max {
            return Err(InterpolateError::OutOfDomain {
                point: xi,
                min: data.x_min,
                max: data.x_max,
                context: data.context.to_string(),
            });
        }

        let idx = find_interval(data.x_data, data.n, xi);
        let point = HermitePoint {
            x0: data.x_data[idx],
            x1: data.x_data[idx + 1],
            y0: data.y_data[idx],
            y1: data.y_data[idx + 1],
            d0: data.slopes[idx],
            d1: data.slopes[idx + 1],
        };

        y_new_data.push(hermite_eval(&point, xi));
    }

    let device = x_new.device();
    Ok(Tensor::from_slice(&y_new_data, &[y_new_data.len()], device))
}

/// Evaluate the derivative of a Hermite interpolant at multiple points.
pub fn derivative_hermite<R: Runtime, C: RuntimeClient<R>>(
    _client: &C,
    x_new: &Tensor<R>,
    data: &HermiteData<'_>,
) -> InterpolateResult<Tensor<R>> {
    let x_new_data: Vec<f64> = x_new.to_vec();
    let mut dy_data = Vec::with_capacity(x_new_data.len());

    for &xi in &x_new_data {
        if xi < data.x_min || xi > data.x_max {
            return Err(InterpolateError::OutOfDomain {
                point: xi,
                min: data.x_min,
                max: data.x_max,
                context: data.context.to_string(),
            });
        }

        let idx = find_interval(data.x_data, data.n, xi);
        let point = HermitePoint {
            x0: data.x_data[idx],
            x1: data.x_data[idx + 1],
            y0: data.y_data[idx],
            y1: data.y_data[idx + 1],
            d0: data.slopes[idx],
            d1: data.slopes[idx + 1],
        };

        dy_data.push(hermite_derivative(&point, xi));
    }

    let device = x_new.device();
    Ok(Tensor::from_slice(&dy_data, &[dy_data.len()], device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_eval_linear() {
        // Linear function y = 2x through (0,0) and (1,2) with slope 2 at both ends
        let point = HermitePoint {
            x0: 0.0,
            x1: 1.0,
            y0: 0.0,
            y1: 2.0,
            d0: 2.0,
            d1: 2.0,
        };

        assert!((hermite_eval(&point, 0.0) - 0.0).abs() < 1e-10);
        assert!((hermite_eval(&point, 0.5) - 1.0).abs() < 1e-10);
        assert!((hermite_eval(&point, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_derivative_linear() {
        // Linear function y = 2x, derivative should be 2 everywhere
        let point = HermitePoint {
            x0: 0.0,
            x1: 1.0,
            y0: 0.0,
            y1: 2.0,
            d0: 2.0,
            d1: 2.0,
        };

        assert!((hermite_derivative(&point, 0.0) - 2.0).abs() < 1e-10);
        assert!((hermite_derivative(&point, 0.5) - 2.0).abs() < 1e-10);
        assert!((hermite_derivative(&point, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_interval() {
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];

        assert_eq!(find_interval(&x_data, 5, 0.0), 0);
        assert_eq!(find_interval(&x_data, 5, 0.5), 0);
        assert_eq!(find_interval(&x_data, 5, 1.0), 1);
        assert_eq!(find_interval(&x_data, 5, 2.5), 2);
        assert_eq!(find_interval(&x_data, 5, 4.0), 3);
    }

    #[test]
    fn test_hermite_passes_through_endpoints() {
        // Arbitrary cubic segment
        let point = HermitePoint {
            x0: 1.0,
            x1: 3.0,
            y0: 5.0,
            y1: 7.0,
            d0: 1.5,
            d1: 0.5,
        };

        // Must pass through endpoints exactly
        assert!((hermite_eval(&point, 1.0) - 5.0).abs() < 1e-10);
        assert!((hermite_eval(&point, 3.0) - 7.0).abs() < 1e-10);

        // Derivative at endpoints must match specified slopes
        assert!((hermite_derivative(&point, 1.0) - 1.5).abs() < 1e-10);
        assert!((hermite_derivative(&point, 3.0) - 0.5).abs() < 1e-10);
    }
}
