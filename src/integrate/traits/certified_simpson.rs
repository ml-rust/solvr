//! Fixed Simpson integration with caller-supplied function and derivative bounds.

use super::certified_simpson_types::{
    CertifiedSimpsonOptions, CertifiedSimpsonPanel, CertifiedSimpsonQuery, CertifiedSimpsonResult,
    CertifiedSimpsonValues,
};
use crate::DType;
use crate::integrate::error::IntegrateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Integrate batches with a fixed panel count and a certified error interval.
///
/// Supply function enclosures and fourth-derivative bounds for each queried interval.
/// The returned guarantee depends on those mathematical bounds.
pub trait CertifiedSimpsonAlgorithms<R: Runtime<DType = DType>> {
    /// Integrate each interval using composite Simpson quadrature.
    ///
    /// # Inputs
    ///
    /// `left` and `right` require matching rank-one shape, F32/F64 dtype, and the configured device.
    /// Endpoints must be finite, with `left <= right`. Strided inputs remain supported.
    /// Zero-width lanes return exact zeros without owner callbacks. Empty batches return empty tensors without callbacks.
    ///
    /// # Callback guarantees
    ///
    /// Callbacks must satisfy these conditions:
    ///
    /// - Function intervals contain every function value throughout each query's coordinate interval.
    /// - Nominal values evaluate the represented query point and lie within their returned function interval.
    /// - Derivative bounds cover the absolute fourth derivative throughout each queried panel interval.
    /// - Functions have four continuous derivatives on each exact closed panel, with one-sided endpoint limits.
    ///
    /// Bounds must cover uncertain parameters. Captured parameters remain stable throughout the call.
    /// Treat query tensors as immutable. Split nonsmooth knots into separate integration calls.
    ///
    /// For `N` panels, exact real nodes satisfy `x_i = left + (right-left)*i/(2*N)`.
    /// Coordinate enclosures contain these nodes, even when represented query points repeat.
    /// `width = (right-left)/N` is the exact mathematical panel width.
    /// Each panel's quadrature error is bounded by `width^5 * fourth_bound / 2880`.
    ///
    /// # Resources and results
    ///
    /// Successful active batches invoke each callback once. I64 owners identify original batch lanes.
    /// Function outputs are copied before the derivative callback runs.
    /// Value queries total `K*(2*N+1)`. Derivative queries total `K*N`, for `K` positive-width lanes.
    /// Their combined count must fit `options.max_points` before query allocation or callback execution.
    /// Batch length, node count, and flattened callback-query counts must fit `u32::MAX` on every backend.
    /// Allocation bytes and index offsets receive separate checks.
    ///
    /// The result separates total absolute error from quadrature error. Inspect its width against your accuracy requirement.
    /// Panel counts remain fixed. The method never adapts, rescales inputs, or retries a different dtype.
    /// Result tensors retain the input shape, dtype, and device, with storage independent of inputs and callback outputs.
    ///
    /// # Arithmetic and errors
    ///
    /// The runtime must declare `FloatArithmeticContract::Ieee754Basic` for the selected dtype.
    /// Admission checks nearest-even rounding, gradual underflow, and Numr TwoSum witnesses on the configured client.
    /// Every executing thread must use the admitted arithmetic mode throughout the call.
    /// Sampled probes do not establish modes for every worker thread.
    /// Provider admission precedes floating input checks, including empty batches.
    /// WebGPU lacks the required arithmetic declaration and returns `IntegrateError::UnsupportedArithmetic`.
    ///
    /// Malformed inputs and callback outputs return input or parameter errors. Callback errors return unchanged.
    /// Nonfinite intermediate bounds return `IntegrateError::ArithmeticRange`. No partial result escapes.
    fn certified_simpson<F, D>(
        &self,
        left: &Tensor<R>,
        right: &Tensor<R>,
        options: CertifiedSimpsonOptions,
        values: F,
        fourth_bound: D,
    ) -> IntegrateResult<CertifiedSimpsonResult<R>>
    where
        F: FnMut(&CertifiedSimpsonQuery<R>) -> IntegrateResult<CertifiedSimpsonValues<R>>,
        D: FnMut(&CertifiedSimpsonPanel<R>) -> IntegrateResult<Tensor<R>>;
}
