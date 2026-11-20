//! Owned root continuation and isolated progress snapshots.

use super::certified_bisection_types::{CertifiedRootProbe, CertifiedRootResult};
use crate::DType;
use crate::optimize::roots::impl_generic::certified_bisection::classification::State;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Opaque numerical state preserving certified root existence across stopping phases.
///
/// Resume consumes this value. Residual functions and derivative assertions stay fixed throughout its lifetime.
pub struct CertifiedRootContinuation<R: Runtime<DType = DType>> {
    pub(crate) state: State<R>,
    pub(crate) floor: Tensor<R>,
    pub(crate) max_iterations: usize,
    pub(crate) max_points: usize,
    pub(crate) iterations: usize,
    pub(crate) mapped_points: usize,
    pub(crate) probe: CertifiedRootProbe,
}

/// Isolated result snapshot and the owned state required for further refinement.
///
/// `Converged` means the requested mapped-coordinate tolerance passes.
pub struct CertifiedMappedRootProgress<R: Runtime<DType = DType>> {
    pub(crate) result: CertifiedRootResult<R>,
    pub(crate) continuation: CertifiedRootContinuation<R>,
}

impl<R: Runtime<DType = DType>> CertifiedMappedRootProgress<R> {
    /// Current outcomes and parameter enclosures, independent of continuation storage.
    pub fn result(&self) -> &CertifiedRootResult<R> {
        &self.result
    }

    /// Cumulative domain points supplied to mapping callbacks, separate from residual work.
    pub fn mapped_points(&self) -> usize {
        self.continuation.mapped_points
    }

    /// Transfer the isolated snapshot and opaque continuation to the caller.
    pub fn into_parts(self) -> (CertifiedRootResult<R>, CertifiedRootContinuation<R>) {
        (self.result, self.continuation)
    }
}
