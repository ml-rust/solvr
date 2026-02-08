//! NURBS (Non-Uniform Rational B-Spline) curve trait definitions.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A NURBS curve defined by weighted control points and knot vector.
#[derive(Debug, Clone)]
pub struct NurbsCurve<R: Runtime> {
    /// Control points, shape [n_points, n_dims].
    pub control_points: Tensor<R>,
    /// Weights for each control point, shape [n_points].
    pub weights: Tensor<R>,
    /// Knot vector, shape [n_knots]. Must be non-decreasing.
    pub knots: Tensor<R>,
    /// Polynomial degree.
    pub degree: usize,
}

/// NURBS curve algorithms.
pub trait NurbsCurveAlgorithms<R: Runtime> {
    /// Evaluate the NURBS curve at parameter values t.
    ///
    /// Uses rational B-spline evaluation: C(t) = sum(w_i * N_i(t) * P_i) / sum(w_i * N_i(t))
    fn nurbs_curve_evaluate(
        &self,
        curve: &NurbsCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate the derivative of the NURBS curve at parameter values t.
    ///
    /// Uses the quotient rule on the rational form.
    fn nurbs_curve_derivative(
        &self,
        curve: &NurbsCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Subdivide the NURBS curve at parameter t.
    fn nurbs_curve_subdivide(
        &self,
        curve: &NurbsCurve<R>,
        t: f64,
    ) -> InterpolateResult<(NurbsCurve<R>, NurbsCurve<R>)>;
}
