//! Cubic spline interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Boundary condition for cubic spline.
#[derive(Debug, Clone, Default)]
pub enum SplineBoundary {
    /// Natural spline: second derivative is zero at endpoints.
    #[default]
    Natural,
    /// Clamped spline: first derivative is specified at endpoints.
    Clamped { left: f64, right: f64 },
    /// Not-a-knot: third derivative is continuous at second and second-to-last points.
    NotAKnot,
}

/// Cubic spline coefficient tensors (a, b, c, d) for polynomial segments.
pub type SplineCoefficients<R> = (Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>);

/// Cubic spline interpolation algorithm.
///
/// Cubic splines provide C2 (continuous second derivative) interpolation
/// with various boundary conditions.
pub trait CubicSplineAlgorithms<R: Runtime<DType = DType>> {
    /// Compute cubic spline coefficients from data points.
    ///
    /// # Arguments
    ///
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    /// * `boundary` - Boundary condition for the spline
    ///
    /// # Returns
    ///
    /// A tuple of (a, b, c, d) coefficient tensors for the spline polynomials.
    fn cubic_spline_coefficients(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        boundary: &SplineBoundary,
    ) -> InterpolateResult<SplineCoefficients<R>>;
}
