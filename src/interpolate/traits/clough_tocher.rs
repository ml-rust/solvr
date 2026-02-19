//! Clough-Tocher 2D interpolator trait.
//!
//! C1-continuous piecewise cubic interpolation on a Delaunay triangulation.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use crate::spatial::traits::delaunay::Delaunay;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A fitted Clough-Tocher 2D interpolator.
///
/// Stores the Delaunay triangulation, vertex values, and estimated gradients.
#[derive(Debug, Clone)]
pub struct CloughTocher2D<R: Runtime<DType = DType>> {
    /// Delaunay triangulation of the input points.
    pub triangulation: Delaunay<R>,
    /// Function values at vertices, shape `[n]`.
    pub values: Tensor<R>,
    /// Estimated gradients at vertices, shape `[n, 2]`.
    pub gradients: Tensor<R>,
    /// Fill value for points outside the convex hull.
    pub fill_value: f64,
}

/// Algorithms for Clough-Tocher C1 piecewise cubic interpolation on 2D scattered data.
///
/// Given scattered points and values, constructs a C1-continuous interpolant
/// using the Clough-Tocher split of each Delaunay triangle.
pub trait CloughTocher2DAlgorithms<R: Runtime<DType = DType>> {
    /// Fit a Clough-Tocher interpolant to scattered 2D data.
    ///
    /// # Arguments
    /// * `points` - 2D tensor of point coordinates, shape `[n, 2]`
    /// * `values` - 1D tensor of values at each point, shape `[n]`
    /// * `fill_value` - Value for query points outside the convex hull (default: NaN)
    fn clough_tocher_fit(
        &self,
        points: &Tensor<R>,
        values: &Tensor<R>,
        fill_value: f64,
    ) -> InterpolateResult<CloughTocher2D<R>>;

    /// Evaluate the Clough-Tocher interpolant at query points.
    ///
    /// # Arguments
    /// * `ct` - The fitted Clough-Tocher interpolator
    /// * `xi` - 2D tensor of query points, shape `[m, 2]`
    ///
    /// # Returns
    /// 1D tensor of interpolated values, shape `[m]`.
    /// Points outside the convex hull get `fill_value`.
    fn clough_tocher_evaluate(
        &self,
        ct: &CloughTocher2D<R>,
        xi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;
}
