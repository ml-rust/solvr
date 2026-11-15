//! Native outcomes for bounded certified upper brackets.

use super::certified_bisection_types::CertifiedRootValues;
use crate::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Stable I64 outcome codes for original input rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub enum CertifiedUpperBracketStatus {
    /// Endpoint signs or an exact singleton zero residual certify root existence.
    Bracketed = 0,
    /// Endpoint bounds certify absence throughout the capped domain.
    NoRoot = 1,
    /// Bounds at the upper limit establish neither root existence nor absence.
    Unresolved = 2,
    /// An expansion or callback-point budget stops a pending row.
    WorkLimit = 3,
}

impl CertifiedUpperBracketStatus {
    /// Decode an outcome code. Unknown codes return `None`.
    pub const fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Self::Bracketed),
            1 => Some(Self::NoRoot),
            2 => Some(Self::Unresolved),
            3 => Some(Self::WorkLimit),
            _ => None,
        }
    }
}

/// Owned endpoint snapshots, residual bounds, and outcomes in original input order.
///
/// Every tensor has shape `[N]`. Endpoint and residual tensors retain the input numerical dtype.
/// Status and expansion tensors use I64. Every row remains present, regardless of its status.
/// A bracket establishes root existence without establishing uniqueness or a requested root accuracy.
/// Keep every returned tensor and its storage aliases immutable.
#[derive(Debug)]
pub struct CertifiedUpperBracketResult<R: Runtime<DType = DType>> {
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
    pub(crate) lower_residual: CertifiedRootValues<R>,
    pub(crate) upper_residual: CertifiedRootValues<R>,
    pub(crate) status: Tensor<R>,
    pub(crate) expansions: Tensor<R>,
    pub(crate) expansion_rounds: usize,
    pub(crate) callback_points: usize,
}

impl<R: Runtime<DType = DType>> CertifiedUpperBracketResult<R> {
    /// Fixed lower endpoints in original input order.
    pub fn lower(&self) -> &Tensor<R> {
        &self.lower
    }

    /// Initial or expanded upper endpoints, bounded by their supplied limits.
    pub fn upper(&self) -> &Tensor<R> {
        &self.upper
    }

    /// Physical residual enclosures at the fixed lower endpoints.
    pub fn lower_residual(&self) -> &CertifiedRootValues<R> {
        &self.lower_residual
    }

    /// Physical residual enclosures at the returned upper endpoints.
    pub fn upper_residual(&self) -> &CertifiedRootValues<R> {
        &self.upper_residual
    }

    /// I64 outcome codes in original input order.
    pub fn status(&self) -> &Tensor<R> {
        &self.status
    }

    /// I64 expansion counts for each original input row.
    pub fn expansions(&self) -> &Tensor<R> {
        &self.expansions
    }

    /// Number of expansion rounds performed across active rows.
    pub fn expansion_rounds(&self) -> usize {
        self.expansion_rounds
    }

    /// Total query points passed to the callback, including both initial endpoints.
    pub fn callback_points(&self) -> usize {
        self.callback_points
    }
}
