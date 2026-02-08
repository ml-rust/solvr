//! B-spline curve trait definitions.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A parametric B-spline curve defined by control points and knot vector.
#[derive(Debug, Clone)]
pub struct BSplineCurve<R: Runtime> {
    /// Control points, shape [n_points, n_dims].
    pub control_points: Tensor<R>,
    /// Knot vector, shape [n_knots]. Must be non-decreasing.
    pub knots: Tensor<R>,
    /// Polynomial degree.
    pub degree: usize,
}

/// B-spline curve algorithms.
pub trait BSplineCurveAlgorithms<R: Runtime> {
    /// Evaluate the B-spline curve at parameter values t.
    ///
    /// # Arguments
    /// * `curve` - The B-spline curve
    /// * `t` - 1D tensor of parameter values, shape [m]
    ///
    /// # Returns
    /// Points on the curve, shape [m, n_dims]
    fn bspline_curve_evaluate(
        &self,
        curve: &BSplineCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate the derivative of the B-spline curve at parameter values t.
    ///
    /// # Arguments
    /// * `order` - Derivative order (1 = first derivative, etc.)
    fn bspline_curve_derivative(
        &self,
        curve: &BSplineCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Subdivide the B-spline curve at parameter t via knot insertion.
    fn bspline_curve_subdivide(
        &self,
        curve: &BSplineCurve<R>,
        t: f64,
    ) -> InterpolateResult<(BSplineCurve<R>, BSplineCurve<R>)>;
}
