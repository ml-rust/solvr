//! Smooth bivariate spline trait — smoothing spline for scattered 2D data.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithms for smoothing bivariate spline fitting on scattered data.
///
/// Fits a tensor-product B-spline surface to scattered (x, y, z) data with
/// optional smoothing. When `smoothing = 0`, interpolates exactly through all
/// points. When `smoothing > 0`, trades off closeness to data against surface
/// smoothness.
pub trait SmoothBivariateSplineAlgorithms<R: Runtime> {
    /// Fit a smoothing bivariate spline to scattered data.
    ///
    /// # Arguments
    /// * `x` - 1D tensor of x coordinates, shape [m]
    /// * `y` - 1D tensor of y coordinates, shape [m]
    /// * `z` - 1D tensor of values, shape [m]
    /// * `weights` - Optional 1D tensor of weights, shape [m]. If None, uniform weights.
    /// * `smoothing` - Smoothing factor (0 = interpolation, >0 = smoothing)
    /// * `kx` - Degree in x direction (typically 3)
    /// * `ky` - Degree in y direction (typically 3)
    #[allow(clippy::too_many_arguments)]
    fn smooth_bivariate_spline_fit(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        z: &Tensor<R>,
        weights: Option<&Tensor<R>>,
        smoothing: f64,
        kx: usize,
        ky: usize,
    ) -> InterpolateResult<BivariateSpline<R>>;

    /// Evaluate a smooth bivariate spline at query points.
    ///
    /// # Arguments
    /// * `spline` - The fitted bivariate spline
    /// * `xi` - 1D tensor of x query coordinates, shape [m]
    /// * `yi` - 1D tensor of y query coordinates, shape [m]
    ///
    /// # Returns
    /// 1D tensor of interpolated values, shape [m]
    fn smooth_bivariate_spline_evaluate(
        &self,
        spline: &BivariateSpline<R>,
        xi: &Tensor<R>,
        yi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>>;
}
