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
    /// 1D tensor of interpolated y values, retaining the input dtype.
    ///
    /// Inputs require matching backend-supported F32/F64 dtypes and the configured client's device.
    /// Grid tensors require rank1, matching lengths, and at least two finite points.
    /// Coordinates require strictly positive finite represented adjacent widths.
    /// Query tensors require rank1 and finite coordinates. Empty queries return an empty tensor after grid checks.
    ///
    /// Linear interpolation uses the unperturbed adjacent width and preserves stored values at exact knots.
    /// Constant intervals return the stored ordinate without coordinate arithmetic.
    /// Outside the grid, Linear extrapolates using the first or last interval.
    /// Nonfinite or underflowed active linear intermediates return an error. Rescale inputs before retrying.
    /// Results otherwise retain ordinary backend rounding, including absorbed increments.
    /// The exact arithmetic envelope requires nearest-even rounding and gradual underflow.
    ///
    /// Nearest and Cubic retain their existing approximate `1e-14` denominator perturbations.
    /// Their method-specific F64 constants and I64 indices remain unchanged.
    /// They do not inherit Linear's exact-knot or portable F32/index guarantees.
    ///
    /// # Errors
    ///
    /// Rejects malformed metadata, nonfinite inputs, and invalid adjacent widths before indexing.
    /// Returns underlying backend errors for unsupported operations or devices.
    fn interp1d(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        x_new: &Tensor<R>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<R>>;
}
