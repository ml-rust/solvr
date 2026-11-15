//! Closed-support Linear interpolation with independent absolute-error receipts.

use crate::DType;
use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Opaque Linear values, exact bracket metadata, and absolute-error enclosures.
///
/// Every tensor retains the query shape and configured device. Values and error bounds retain
/// the matching F32/F64 input dtype. Indices retain the backend's existing search index dtype.
#[derive(Debug, Clone)]
pub struct LinearInterp1dReceipt<R: Runtime<DType = DType>> {
    pub(crate) values: Tensor<R>,
    pub(crate) absolute_error: Tensor<R>,
    pub(crate) lower_indices: Tensor<R>,
    pub(crate) upper_indices: Tensor<R>,
    pub(crate) at_lower: Tensor<R>,
    pub(crate) at_upper: Tensor<R>,
    pub(crate) constant: Tensor<R>,
}
impl<R: Runtime<DType = DType>> LinearInterp1dReceipt<R> {
    /// Same successful values as ordinary matching-dtype Linear interpolation on this backend/layout.
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }
    /// Finite nonnegative upper bounds against exact original-input Linear interpolation.
    ///
    /// Bounds describe an exact real enclosure around each represented value. No rounded
    /// value-minus-bound or value-plus-bound endpoints are formed. Exact knots and holds return zero.
    pub fn absolute_error(&self) -> &Tensor<R> {
        &self.absolute_error
    }
    /// Existing selected lower bracket indices, retaining the backend's index dtype.
    pub fn lower_indices(&self) -> &Tensor<R> {
        &self.lower_indices
    }
    /// Existing selected upper bracket indices, retaining the backend's index dtype.
    pub fn upper_indices(&self) -> &Tensor<R> {
        &self.upper_indices
    }
    /// Exact original query-to-lower-knot equality masks in the input floating dtype.
    pub fn at_lower(&self) -> &Tensor<R> {
        &self.at_lower
    }
    /// Exact original query-to-upper-knot equality masks in the input floating dtype.
    pub fn at_upper(&self) -> &Tensor<R> {
        &self.at_upper
    }
    /// Exact selected bracket-ordinate equality masks in the input floating dtype.
    pub fn constant(&self) -> &Tensor<R> {
        &self.constant
    }
}

/// Independent absolute-error receipts for matching F32/F64 Linear interpolation under declared IEEE basic operations.
///
/// Existing interpolation traits and their implementation requirements remain unchanged.
pub trait LinearInterp1dDiagnostics<R: Runtime<DType = DType>> {
    /// Evaluate Linear interpolation within the closed original grid domain.
    ///
    /// Inputs require the ordinary rank1, matching-dtype/device, finite-data, and increasing-grid checks.
    /// Queries outside the original domain reject. Empty queries retain empty metadata after grid checks.
    /// Exact knots preserve the existing left-bracket convention at interior knots. Constant brackets hold exactly.
    ///
    /// After metadata checks, the runtime must declare `FloatArithmeticContract::Ieee754Basic` for the selected dtype.
    /// Backend-defined providers return `UnsupportedArithmetic` with `failed_probes = ["provider_contract"]` before floating checks.
    /// This static provider requirement is independent of sampled device behavior. WebGPU retains backend-defined arithmetic.
    ///
    /// Each admitted provider call then checks nearest-even rounding, gradual input/output underflow, and representative Numr TwoSum witnesses.
    /// One combined scalar admission synchronization applies on the actual client, device, and dtype.
    /// These probes supplement Numr's primitive rounding contract. Execution mode must remain stable throughout the call.
    /// No operand-array readback, CPU fallback, extrapolation receipt, rescaling, or relative-error guarantee applies.
    ///
    /// The receipt retains TwoSum lows for grid width, query offset, and signed ordinate difference.
    /// Their high components must match the ordinary evaluation's represented intermediates.
    /// For active interior lanes, outward arithmetic bounds the exact normalized-coordinate error and signed-value error:
    ///
    /// ```text
    /// coordinate_error >= (abs(width_low)+abs(offset_low))/width_high + B(coordinate)
    /// absolute_error >= abs(ordinate_low) + abs(ordinate_high)*coordinate_error
    ///                   + B(increment) + B(value)
    /// B(z) = fl(fl(dtype_epsilon*abs(z)) + dtype_minimum_subnormal)
    /// ```
    ///
    /// Every positive uncertainty addition, product, and division rounds outward. Exact zero terms retain zero.
    /// A positive underflowed uncertainty contribution retains outward padding. No unit-scale tolerance floor applies.
    /// Successful values retain the ordinary operation graph, including cancellation to zero and absorbed increments.
    /// A finite ordinary value with an unrepresentable receipt rejects only this diagnostic call.
    ///
    /// # Errors
    ///
    /// Returns ordinary input/backend errors for unsupported dtypes or malformed inputs.
    /// Missing provider contracts and detected arithmetic-envelope mismatches return `InterpolateError::UnsupportedArithmetic`.
    /// Nonfinite receipt arithmetic returns a numerical error without changing the ordinary interpolation API.
    fn interp1d_linear_with_receipts(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        query: &Tensor<R>,
    ) -> InterpolateResult<LinearInterp1dReceipt<R>>;
}
