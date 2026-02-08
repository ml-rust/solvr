//! Bezier surface trait definitions.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A Bezier surface defined by a 2D grid of control points.
#[derive(Debug, Clone)]
pub struct BezierSurface<R: Runtime> {
    /// Control points, shape [nu, nv, n_dims].
    pub control_points: Tensor<R>,
    /// Degree in u direction (nu - 1).
    pub degree_u: usize,
    /// Degree in v direction (nv - 1).
    pub degree_v: usize,
}

/// Bezier surface algorithms.
pub trait BezierSurfaceAlgorithms<R: Runtime> {
    /// Evaluate the Bezier surface at parameter values (u, v) in [0, 1]^2.
    ///
    /// # Arguments
    /// * `u` - 1D tensor of u parameter values, shape [m]
    /// * `v` - 1D tensor of v parameter values, shape [m]
    ///
    /// # Returns
    /// Points on the surface, shape [m, n_dims]
    fn bezier_surface_evaluate(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate partial derivatives of the Bezier surface.
    ///
    /// # Arguments
    /// * `du` - Derivative order in u direction
    /// * `dv` - Derivative order in v direction
    fn bezier_surface_partial(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
        du: usize,
        dv: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Compute surface normals at parameter values (u, v).
    fn bezier_surface_normal(
        &self,
        surface: &BezierSurface<R>,
        u: &Tensor<R>,
        v: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;
}
