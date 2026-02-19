//! Rect bivariate spline trait — tensor-product B-spline on rectangular grid.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::traits::bspline::BSplineBoundary;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A fitted tensor-product bivariate B-spline on a rectangular grid.
///
/// Represents `S(x,y) = Σᵢ Σⱼ cᵢⱼ Bᵢ(x) Bⱼ(y)` where Bᵢ, Bⱼ are 1D B-spline
/// basis functions along x and y axes respectively.
#[derive(Debug, Clone)]
pub struct BivariateSpline<R: Runtime<DType = DType>> {
    /// Knot vector for x-axis, shape `[n_knots_x]`.
    pub knots_x: Tensor<R>,
    /// Knot vector for y-axis, shape `[n_knots_y]`.
    pub knots_y: Tensor<R>,
    /// Coefficient matrix, shape `[n_coeffs_x, n_coeffs_y]`.
    pub coefficients: Tensor<R>,
    /// Polynomial degree along x-axis.
    pub degree_x: usize,
    /// Polynomial degree along y-axis.
    pub degree_y: usize,
}

/// Algorithms for tensor-product B-spline interpolation on rectangular grids.
///
/// Given a regular grid of (x, y) values and corresponding z values,
/// fits a smooth bivariate B-spline surface. Supports evaluation,
/// partial derivatives, and integration over rectangular domains.
pub trait RectBivariateSplineAlgorithms<R: Runtime<DType = DType>> {
    /// Fit a tensor-product B-spline to data on a rectangular grid.
    ///
    /// # Arguments
    /// * `x` - 1D tensor of x coordinates, shape `[nx]`, strictly increasing
    /// * `y` - 1D tensor of y coordinates, shape `[ny]`, strictly increasing
    /// * `z` - 2D tensor of values, shape `[nx, ny]`
    /// * `degree_x` - Polynomial degree along x (typically 3)
    /// * `degree_y` - Polynomial degree along y (typically 3)
    /// * `boundary` - Boundary condition for knot vector construction
    #[allow(clippy::too_many_arguments)]
    fn rect_bivariate_spline_fit(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        z: &Tensor<R>,
        degree_x: usize,
        degree_y: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BivariateSpline<R>>;

    /// Evaluate a bivariate spline at query points.
    ///
    /// # Arguments
    /// * `spline` - The fitted bivariate spline
    /// * `xi` - 1D tensor of x query coordinates, shape `[m]`
    /// * `yi` - 1D tensor of y query coordinates, shape `[m]`
    ///
    /// # Returns
    /// 1D tensor of interpolated values, shape `[m]`
    fn rect_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<R>,
        xi: &Tensor<R>,
        yi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate a bivariate spline on a grid of query points.
    ///
    /// # Arguments
    /// * `spline` - The fitted bivariate spline
    /// * `xi` - 1D tensor of x grid coordinates, shape `[mx]`
    /// * `yi` - 1D tensor of y grid coordinates, shape `[my]`
    ///
    /// # Returns
    /// 2D tensor of interpolated values, shape `[mx, my]`
    fn rect_bivariate_spline_evaluate_grid(
        &self,
        spline: &BivariateSpline<R>,
        xi: &Tensor<R>,
        yi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;

    /// Evaluate partial derivative of a bivariate spline at query points.
    ///
    /// # Arguments
    /// * `spline` - The fitted bivariate spline
    /// * `xi` - 1D tensor of x query coordinates, shape `[m]`
    /// * `yi` - 1D tensor of y query coordinates, shape `[m]`
    /// * `dx` - Derivative order in x direction
    /// * `dy` - Derivative order in y direction
    fn rect_bivariate_spline_partial_derivative(
        &self,
        spline: &BivariateSpline<R>,
        xi: &Tensor<R>,
        yi: &Tensor<R>,
        dx: usize,
        dy: usize,
    ) -> InterpolateResult<Tensor<R>>;

    /// Integrate the bivariate spline over a rectangular domain [xa, xb] × [ya, yb].
    fn rect_bivariate_spline_integrate(
        &self,
        spline: &BivariateSpline<R>,
        xa: f64,
        xb: f64,
        ya: f64,
        yb: f64,
    ) -> InterpolateResult<Tensor<R>>;
}
