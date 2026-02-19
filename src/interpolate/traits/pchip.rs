//! PCHIP interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) algorithm.
///
/// PCHIP is a shape-preserving interpolation method that preserves monotonicity
/// and avoids overshoot. It has continuous first derivative (C1).
pub trait PchipAlgorithms<R: Runtime<DType = DType>> {
    /// Compute PCHIP slopes from data points using Fritsch-Carlson method.
    ///
    /// # Arguments
    ///
    /// * `x` - 1D tensor of x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values (same length as x)
    ///
    /// # Returns
    ///
    /// Tensor of slopes at each point that preserve monotonicity.
    fn pchip_slopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>>;
}
