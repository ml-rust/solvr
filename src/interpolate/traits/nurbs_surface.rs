//! NURBS surface trait definitions.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A NURBS surface defined by weighted control points and two knot vectors.
#[derive(Debug, Clone)]
pub struct NurbsSurface<R: Runtime<DType = DType>> {
    /// Control points, shape `[nu, nv, n_dims]`.
    pub control_points: Tensor<R>,
    /// Weights, shape `[nu, nv]`.
    pub weights: Tensor<R>,
    /// Knot vector in u direction, shape `[n_knots_u]`.
    pub knots_u: Tensor<R>,
    /// Knot vector in v direction, shape `[n_knots_v]`.
    pub knots_v: Tensor<R>,
    /// Degree in u direction.
    pub degree_u: usize,
    /// Degree in v direction.
    pub degree_v: usize,
}

/// NURBS surface algorithms.
pub trait NurbsSurfaceAlgorithms<R: Runtime<DType = DType>> {
    /// Evaluate the NURBS surface at parameter values (u, v).
    fn nurbs_surface_evaluate(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate partial derivatives of the NURBS surface.
    fn nurbs_surface_partial(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Compute surface normals at parameter values (u, v).
    fn nurbs_surface_normal(
        &self,
        surface: &NurbsSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;
}
