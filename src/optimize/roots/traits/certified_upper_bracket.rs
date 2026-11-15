//! Bounded upper expansion for increasing physical residuals.

use super::certified_bisection_types::{CertifiedRootQuery, CertifiedRootValues};
use super::certified_upper_bracket_types::CertifiedUpperBracketResult;
use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Expansion and callback work limits for independent upper brackets.
#[derive(Clone, Debug)]
pub struct CertifiedUpperBracketOptions {
    /// Maximum expansion rounds, at most `i64::MAX`. Zero permits initial endpoint classification only.
    pub max_expansions: usize,
    /// Maximum callback points, including the initial `2*N` admission budget.
    pub max_points: usize,
}

impl Default for CertifiedUpperBracketOptions {
    fn default() -> Self {
        Self {
            max_expansions: 4096,
            max_points: 1 << 20,
        }
    }
}

/// Find bounded upper endpoints that certify root existence with fixed lower endpoints.
pub trait CertifiedUpperBracketAlgorithms<R: Runtime<DType = DType>> {
    /// Expand upper endpoints toward their limits using physical residual enclosures.
    ///
    /// Each row describes one fixed continuous nondecreasing function throughout its capped domain.
    /// This operation establishes a bracket. It neither contracts roots nor establishes root uniqueness.
    /// Inputs have matching rank-one F32 or F64 shapes, dtypes, and devices.
    /// Broadcasting does not apply. Every row satisfies finite `lower <= initial_upper <= upper_limit`.
    /// Lower endpoints stay fixed. Returned upper endpoints stay between their initial values and limits.
    ///
    /// Callback bounds enclose the physical function at each exact represented query point.
    /// Bounds match `query.point` shape, dtype, and device. Bounds are finite and ordered.
    /// Original I64 owner indices survive active-row compaction.
    /// Keep query tensors and retained storage aliases immutable until the operation returns.
    /// Inputs and callback bounds receive native snapshots before another callback can overwrite shared storage.
    ///
    /// `f(lower).upper <= 0` and `f(upper).lower >= 0` together certify a bracket.
    /// An exact `[0, 0]` residual at either endpoint also certifies a bracket.
    /// Bounds merely touching or overlapping zero never collapse an interval.
    /// A strictly positive residual lower bound at the lower endpoint certifies absence.
    /// A strictly negative residual upper bound at the upper limit also certifies absence.
    /// Negative or uncertain intermediate upper endpoints require expansion.
    /// Unclassified rows at their upper limits return `Unresolved`.
    /// Pending rows stopped by expansion or callback budgets return `WorkLimit`.
    ///
    /// Every executing thread preserves nearest-even rounding and gradual underflow.
    /// Unsupported providers and execution modes reject the call without fallback.
    /// Provider and execution-mode checks run even for empty inputs. Empty inputs invoke no callback.
    /// Initial work reserves and evaluates `2*N` callback points, including immediately classified rows.
    /// A smaller `max_points` returns `InvalidParameter` before invoking the callback.
    /// Callback errors, malformed inputs or bounds, and arithmetic errors reject the entire call.
    /// Keep returned tensors and their storage aliases immutable. Discard partial results after an error.
    fn bracket_upper_certified<F>(
        &self,
        f: F,
        lower: &Tensor<R>,
        initial_upper: &Tensor<R>,
        upper_limit: &Tensor<R>,
        options: &CertifiedUpperBracketOptions,
    ) -> OptimizeResult<CertifiedUpperBracketResult<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>;
}
