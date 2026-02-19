//! Akima interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Akima 1D interpolation algorithm.
///
/// Akima interpolation is a locally-weighted cubic interpolation method that
/// is robust to outliers. It has continuous first derivative (C1).
pub trait AkimaAlgorithms<R: Runtime<DType = DType>> {
    /// Compute Akima interpolator from data points.
    ///
    /// # Arguments
    ///
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    ///
    /// # Returns
    ///
    /// A tuple of (slopes, x_min, x_max) where slopes are the computed slopes
    /// at each point, and x_min/x_max are the domain bounds.
    fn akima_slopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>>;
}
