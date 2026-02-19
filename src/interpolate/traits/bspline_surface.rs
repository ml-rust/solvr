//! B-spline surface trait definitions.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A B-spline surface defined by a 2D grid of control points and two knot vectors.
#[derive(Debug, Clone)]
pub struct BSplineSurface<R: Runtime<DType = DType>> {
    /// Control points, shape `[nu, nv, n_dims]`.
    pub control_points: Tensor<R>,
    /// Knot vector in u direction, shape `[n_knots_u]`.
    pub knots_u: Tensor<R>,
    /// Knot vector in v direction, shape `[n_knots_v]`.
    pub knots_v: Tensor<R>,
    /// Degree in u direction.
    pub degree_u: usize,
    /// Degree in v direction.
    pub degree_v: usize,
}

/// B-spline surface algorithms.
pub trait BSplineSurfaceAlgorithms<R: Runtime<DType = DType>> {
    /// Evaluate the B-spline surface at parameter values (u, v).
    fn bspline_surface_evaluate(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate partial derivatives of the B-spline surface.
    fn bspline_surface_partial(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Compute surface normals at parameter values (u, v).
    fn bspline_surface_normal(
        &self,
        surface: &BSplineSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;
}
