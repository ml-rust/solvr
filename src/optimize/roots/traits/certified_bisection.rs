//! Increasing roots from physical residual bounds.

use super::certified_bisection_types::{
    CertifiedRootQuery, CertifiedRootResult, CertifiedRootValues,
};
use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Width tolerances and callback work limits for independent roots.
#[derive(Clone, Debug)]
pub struct CertifiedBisectionOptions {
    /// Finite nonnegative absolute enclosure-width allowance.
    pub atol: f64,
    /// Finite nonnegative relative allowance times the largest absolute endpoint.
    pub rtol: f64,
    /// Maximum midpoint rounds. Zero permits endpoint classification only.
    pub max_iterations: usize,
    /// Maximum callback points, including the initial `2*N` admission budget.
    pub max_points: usize,
}

impl Default for CertifiedBisectionOptions {
    fn default() -> Self {
        Self {
            atol: 1e-12,
            rtol: 1e-12,
            max_iterations: 256,
            max_points: 1 << 20,
        }
    }
}

/// Enclose independent increasing roots through the configured native client.
pub trait CertifiedBisectionAlgorithms<R: Runtime<DType = DType>> {
    /// Solve physical equations using residual enclosures and positive slope lower bounds.
    ///
    /// Inputs have matching rank-one F32 or F64 shapes, dtypes, and devices. Broadcasting does not apply.
    /// Endpoints are finite and ordered. Each `derivative_lower` value is finite and strictly positive.
    /// Every row describes one fixed continuous function throughout its supplied domain.
    /// The caller establishes this inequality for every `x < y` in that domain:
    ///
    /// ```text
    /// f(y) - f(x) >= derivative_lower * (y - x)
    /// ```
    ///
    /// Callback bounds enclose the physical function at each exact represented query point.
    /// Callback bounds match `query.point` shape, dtype, and device. Bounds are finite and ordered.
    /// Original I64 owner indices survive active-row compaction.
    /// Keep query tensors and retained storage aliases immutable until the solver returns.
    /// Inputs and callback bounds receive native snapshots before another callback can overwrite shared storage.
    ///
    /// Certified endpoint signs or exact singleton zero residuals establish root existence.
    /// Overlapping zero bounds alone do not establish existence. Nominal zeros never collapse an enclosure.
    /// Width admission uses `atol + rtol * max(abs(lower), abs(upper))`, rounded inward.
    /// Zero absolute and relative tolerances require an exact-width result.
    /// F32 scalar tolerances round downward from their original f64 values.
    ///
    /// Every executing thread preserves nearest-even rounding and gradual underflow.
    /// Provider and execution-mode checks run even for empty input. Empty inputs invoke no callback.
    /// Initial work admission reserves `2*N` callback points, including rows resolved at one endpoint.
    /// Arithmetic limits and malformed inputs or callback bounds reject the entire call.
    /// Keep returned tensors and their storage aliases immutable. Discard partial results after an error.
    fn bisect_certified<F>(
        &self,
        f: F,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        derivative_lower: &Tensor<R>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedRootResult<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>;
}
