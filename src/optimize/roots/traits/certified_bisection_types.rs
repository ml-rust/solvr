//! Native queries, residual bounds, and certified-root outcomes.

use crate::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Probe placement inside a certified root bracket. Probes never certify signs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CertifiedRootProbe {
    /// Bisection midpoint probes.
    #[default]
    Midpoint,
    /// Secant probes gated on a full round of slack against bisection's width schedule.
    ///
    /// A round runs when its midpoints fit the remaining points. Paired second points
    /// join in row order while they fit after those midpoints. The rest are dropped for that round.
    /// A run whose unbounded point count fits `max_points` matches the unbounded run bit
    /// for bit. A binding limit can run fewer rounds than `Midpoint`. Its rows still
    /// report `WorkLimit`, and their brackets stay certified.
    SlackSecant,
}

/// Native callback points whose tensors and retained storage aliases stay immutable until the solver returns.
#[derive(Debug)]
pub struct CertifiedRootQuery<R: Runtime<DType = DType>> {
    /// Original input indices as a rank-one I64 tensor.
    pub owner: Tensor<R>,
    /// Exact represented query points, matching the input numerical dtype.
    pub point: Tensor<R>,
}

/// Finite ordered enclosures returned by the physical residual callback.
#[derive(Debug)]
pub struct CertifiedRootValues<R: Runtime<DType = DType>> {
    /// Residual lower bounds, matching the query point metadata.
    pub lower: Tensor<R>,
    /// Residual upper bounds, matching the query point metadata.
    pub upper: Tensor<R>,
}

/// Stable I64 outcome codes for original input rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub enum CertifiedRootStatus {
    /// A unique root exists and its enclosure meets the requested width.
    Converged = 0,
    /// The supplied closed domain contains no root.
    NoRoot = 1,
    /// Endpoint bounds establish neither root existence nor absence.
    UnresolvedBracket = 2,
    /// A root exists, but this oracle and arithmetic produce no further represented contraction.
    AccuracyFloor = 3,
    /// A root exists, but an iteration or callback-point budget stops further work.
    WorkLimit = 4,
}

impl CertifiedRootStatus {
    /// Decode an outcome code. Unknown codes return `None`.
    pub const fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Self::Converged),
            1 => Some(Self::NoRoot),
            2 => Some(Self::UnresolvedBracket),
            3 => Some(Self::AccuracyFloor),
            4 => Some(Self::WorkLimit),
            _ => None,
        }
    }
}

/// Native outcomes and compact enclosures for rows with certified root existence.
///
/// Root arrays include `Converged`, `AccuracyFloor`, and `WorkLimit` rows only.
/// `NoRoot` and `UnresolvedBracket` rows never appear in those arrays.
/// Accuracy floors do not prove that another oracle or method cannot improve the bounds.
/// Keep every returned tensor and its storage aliases immutable.
#[derive(Debug)]
pub struct CertifiedRootResult<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) root_owner: Tensor<R>,
    pub(crate) root_lower: Tensor<R>,
    pub(crate) root_upper: Tensor<R>,
    pub(crate) iterations: usize,
    pub(crate) evaluated_points: usize,
}

impl<R: Runtime<DType = DType>> CertifiedRootResult<R> {
    /// I64 outcome codes with shape `[N]`, in original input order.
    pub fn status(&self) -> &Tensor<R> {
        &self.status
    }

    /// Original I64 indices with shape `[M]`, identifying the compact root arrays.
    pub fn root_owner(&self) -> &Tensor<R> {
        &self.root_owner
    }

    /// Certified lower endpoints with shape `[M]` and the input numerical dtype.
    pub fn root_lower(&self) -> &Tensor<R> {
        &self.root_lower
    }

    /// Certified upper endpoints with shape `[M]` and the input numerical dtype.
    pub fn root_upper(&self) -> &Tensor<R> {
        &self.root_upper
    }

    /// Number of midpoint rounds performed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Total query points passed to the callback.
    pub fn evaluated_points(&self) -> usize {
        self.evaluated_points
    }
}
