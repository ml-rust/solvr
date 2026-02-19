//! B-spline interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A B-spline curve represented by its knot vector and coefficients.
#[derive(Debug, Clone)]
pub struct BSpline<R: Runtime<DType = DType>> {
    /// Knot vector, shape `[n_knots]`. Must be non-decreasing.
    pub knots: Tensor<R>,
    /// Spline coefficients, shape `[n_coeffs]` for 1D values.
    pub coefficients: Tensor<R>,
    /// Polynomial degree (order = degree + 1).
    pub degree: usize,
}

/// Boundary condition for B-spline construction.
#[derive(Debug, Clone, Default)]
pub enum BSplineBoundary {
    /// Not-a-knot: default, no additional constraints.
    #[default]
    NotAKnot,
    /// Clamped: first derivative specified at endpoints.
    Clamped { left: f64, right: f64 },
    /// Natural: second derivative is zero at endpoints.
    Natural,
}

/// B-spline interpolation algorithms.
///
/// Provides construction, evaluation, differentiation, integration, and
/// root-finding for B-spline curves.
pub trait BSplineAlgorithms<R: Runtime<DType = DType>> {
    /// Construct an interpolating B-spline from data points.
    ///
    /// Builds a B-spline of the given degree that passes through all (x, y) points.
    /// Uses a collocation matrix solved via dense linear algebra.
    ///
    /// # Arguments
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    /// * `degree` - Polynomial degree (typically 3 for cubic)
    /// * `boundary` - Boundary condition
    fn make_interp_spline(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        degree: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BSpline<R>>;

    /// Evaluate a B-spline at new points.
    fn bspline_evaluate(
        &self,
        spline: &BSpline<R>,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate the derivative of a B-spline at new points.
    ///
    /// # Arguments
    /// * `order` - Derivative order (1 = first derivative, 2 = second, etc.)
    fn bspline_derivative(
        &self,
        spline: &BSpline<R>,
        x_new: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Compute the definite integral of a B-spline over [a, b].
    fn bspline_integrate(
        &self,
        spline: &BSpline<R>,
        a: f64,
        b: f64,
    ) -> InterpolateResult<Tensor<R>>;
}
