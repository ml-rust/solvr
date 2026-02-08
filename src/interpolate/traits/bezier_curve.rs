//! Bezier curve trait definitions.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A Bezier curve defined by control points.
#[derive(Debug, Clone)]
pub struct BezierCurve<R: Runtime> {
    /// Control points, shape [n_points, n_dims].
    pub control_points: Tensor<R>,
    /// Polynomial degree (n_points - 1).
    pub degree: usize,
}

/// Bezier curve algorithms.
pub trait BezierCurveAlgorithms<R: Runtime> {
    /// Evaluate the Bezier curve at parameter values t in [0, 1].
    ///
    /// # Arguments
    /// * `curve` - The Bezier curve
    /// * `t` - 1D tensor of parameter values, shape [m]
    ///
    /// # Returns
    /// Points on the curve, shape [m, n_dims]
    fn bezier_evaluate(
        &self,
        curve: &BezierCurve<R>,
        t: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate the derivative of the Bezier curve at parameter values t.
    ///
    /// # Arguments
    /// * `order` - Derivative order (1 = first derivative, etc.)
    fn bezier_derivative(
        &self,
        curve: &BezierCurve<R>,
        t: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Subdivide the Bezier curve at parameter t using de Casteljau's algorithm.
    ///
    /// Returns (left, right) curves where left covers [0, t] and right covers [t, 1].
    fn bezier_subdivide(
        &self,
        curve: &BezierCurve<R>,
        t: f64,
    ) -> InterpolateResult<(BezierCurve<R>, BezierCurve<R>)>;
}
