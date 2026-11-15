//! Certified increasing roots with caller-defined convergence coordinates.

use super::{
    CertifiedBisectionOptions, CertifiedMappedRootProgress, CertifiedRootAdvanceAllowance,
    CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
    CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
};
use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Refine certified roots while retaining numerical state across mapped stopping phases.
pub trait CertifiedMappedBisectionAlgorithms<R: Runtime<DType = DType>> {
    /// Establish roots and stop when complete mapped bounds meet the supplied tolerances.
    ///
    /// Residual inputs and assertions follow `CertifiedBisectionAlgorithms::bisect_certified`.
    /// Mapping bounds enclose the mapped quantity throughout each supplied parameter domain.
    /// They match the query's rank-one shape, numerical dtype, and device, and remain finite and ordered.
    /// No monotonic mapping assumption applies. Exact parameter roots still require mapped convergence.
    /// Query and callback storage aliases stay immutable until the call returns.
    /// Returned snapshots do not alias continuation state.
    /// Initial `2*N` admission and execution qualification also apply to empty inputs.
    fn bisect_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        derivative_lower: &Tensor<R>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<CertifiedMappedRootProgress<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
        M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>;

    /// Resume existing root domains without repeating endpoint admission or resetting counters.
    ///
    /// Residual functions, derivative assertions, enclosure policy, and execution qualification remain fixed throughout the session.
    /// The mapping can change. Its bounds retain the complete-domain enclosure obligation.
    /// Allowances limit additional work and never increase the immutable original ceilings.
    /// Previously converged rows can resume under stricter criteria.
    /// Arithmetic floors retain their domains and can satisfy looser criteria without residual work.
    /// An error consumes the continuation. Discard partial outputs.
    fn resume_certified_mapped<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<R>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
        M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>;
    /// Resume only unique original input-row owners on the configured device.
    ///
    /// Owners are rank-one I64, in range of the original status tensor.
    /// Nonascending owners are valid. Duplicate owners are rejected before callbacks.
    /// Empty demand performs no callback work and preserves the complete continuation.
    /// Undemanded tensors retain every stored bit.
    /// Work counters remain cumulative across the complete continuation.
    /// Owner storage aliases stay immutable until the call returns.
    /// Undemanded Converged status retains its previous stopping criteria.
    /// The new tolerance applies only to demanded rows.
    /// Original work ceilings and execution qualification remain unchanged.
    fn resume_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<R>,
        owners: &Tensor<R>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
        M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>;
    /// Refresh selected roots after changing only the residual enclosure precision policy.
    ///
    /// The exact residual function, coordinate map, derivative assertion, and execution qualification remain fixed.
    /// Owners follow selected-resume admission. Original ceilings and cumulative counters remain unchanged.
    /// Demanded nonsingleton roots discard arithmetic-floor and sampled-midpoint history before iteration.
    /// Exact singleton history and every undemanded field remain unchanged.
    /// The new callback bounds enclose the same residual at every supplied point.
    /// Empty demand performs no callbacks. An error consumes the continuation.
    fn refresh_certified_mapped_selected<F, M>(
        &self,
        residual: F,
        mapping: M,
        continuation: CertifiedRootContinuation<R>,
        owners: &Tensor<R>,
        tolerance: &CertifiedRootTolerance,
        allowance: &CertifiedRootAdvanceAllowance,
    ) -> OptimizeResult<CertifiedMappedRootProgress<R>>
    where
        F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
        M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>;
}
