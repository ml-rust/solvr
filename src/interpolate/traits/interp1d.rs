//! 1D interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Interpolation method for 1D data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation between adjacent points.
    Linear,
    /// Cubic interpolation using 4 neighboring points.
    Cubic,
}

/// 1D interpolation algorithm.
///
/// Provides multiple interpolation methods: nearest neighbor, linear, and cubic.
pub trait Interp1dAlgorithms<R: Runtime<DType = DType>> {
    /// Evaluate 1D interpolation at new x coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - 1D tensor of known x coordinates (must be strictly increasing)
    /// * `y` - 1D tensor of y values at known points
    /// * `x_new` - 1D tensor of x coordinates to interpolate at
    /// * `method` - Interpolation method to use
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated y values.
    fn interp1d(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        x_new: &Tensor<R>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<R>>;
}
