//! Two-component centers with certified absolute error bounds.

use crate::DType;
use numr::ops::ExpansionOperand;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Owned components enclosing real values elementwise in `[high + low - abs_error, high + low + abs_error]`.
///
/// Create values through `IntervalContext::expansion_point` or `IntervalContext::expansion_bounds`.
/// The center sum uses mathematical addition. The radius is finite and nonnegative.
/// Constructors normalize the center with TwoSum while preserving its exact sum.
/// Rows with a zero low component retain the original high and low bit patterns.
///
/// Components and binary operands require identical shapes, dtypes, and devices. No implicit broadcasting applies.
/// Any tensor rank is supported within native view and capture limits.
///
/// Inputs snapshot external storage. Keep accessor tensors and all their storage aliases immutable.
/// Results remain provisional until the enclosing arithmetic call returns `Ok`.
/// Discard partial results after an error.
#[derive(Debug)]
pub struct CertifiedExpansion<R: Runtime<DType = DType>> {
    pub(crate) high: Tensor<R>,
    pub(crate) low: Tensor<R>,
    pub(crate) abs_error: Tensor<R>,
}

impl<R: Runtime<DType = DType>> CertifiedExpansion<R> {
    /// High center component. Keep this tensor and its storage aliases immutable.
    pub fn high(&self) -> &Tensor<R> {
        &self.high
    }

    /// Low center component. Keep this tensor and its storage aliases immutable.
    pub fn low(&self) -> &Tensor<R> {
        &self.low
    }

    /// Nonnegative absolute error radius. Keep this tensor and its storage aliases immutable.
    pub fn abs_error(&self) -> &Tensor<R> {
        &self.abs_error
    }

    /// Borrow the components as a kernel operand.
    pub(crate) fn operand(&self) -> ExpansionOperand<'_, R> {
        ExpansionOperand {
            high: &self.high,
            low: &self.low,
            abs_error: &self.abs_error,
        }
    }
}
