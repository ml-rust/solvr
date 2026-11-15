//! Query tensors and error bounds for certified fixed Simpson integration.

use crate::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Explicit panel count and total callback-query budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertifiedSimpsonOptions {
    /// Positive power-of-two count of whole Simpson panels per active lane.
    ///
    /// F32 requires `2*panels <= 2^24`. Both dtypes require `2*panels+1 <= u32::MAX`.
    pub panels: usize,
    /// Maximum combined function-point and derivative-panel queries across all active lanes.
    ///
    /// The required count is `K*(3*panels+1)`, for `K` positive-width lanes.
    /// This bounds scheduled queries, rather than bytes. Allocation sizes receive separate checks.
    pub max_points: usize,
}

/// Function points and enclosures of the corresponding exact mathematical nodes.
///
/// All tensors have flattened shape `[P]`. Floating tensors retain the endpoint dtype and configured device.
#[derive(Debug, Clone)]
pub struct CertifiedSimpsonQuery<R: Runtime<DType = DType>> {
    /// I64 indices of original batch lanes, before active-lane compaction.
    pub owner: Tensor<R>,
    /// Represented coordinates for nominal function evaluation.
    pub point: Tensor<R>,
    /// Lower endpoints enclosing the exact mathematical nodes and represented points.
    pub lower: Tensor<R>,
    /// Upper endpoints enclosing the exact mathematical nodes and represented points.
    pub upper: Tensor<R>,
}

/// Closed panel intervals requiring absolute fourth-derivative bounds.
///
/// All tensors have flattened shape `[P]`. Each interval contains its entire exact Simpson panel.
#[derive(Debug, Clone)]
pub struct CertifiedSimpsonPanel<R: Runtime<DType = DType>> {
    /// I64 indices of original batch lanes, before active-lane compaction.
    pub owner: Tensor<R>,
    /// Lower panel endpoints in the endpoint dtype and configured device.
    pub lower: Tensor<R>,
    /// Upper panel endpoints in the endpoint dtype and configured device.
    pub upper: Tensor<R>,
}

/// Nominal function values and mathematical enclosures returned by the value callback.
///
/// Every tensor matches the function query's shape, dtype, and device.
#[derive(Debug, Clone)]
pub struct CertifiedSimpsonValues<R: Runtime<DType = DType>> {
    /// Function evaluation at each represented query point, inside its returned interval.
    pub value: Tensor<R>,
    /// Lower bounds for every function value throughout each coordinate interval.
    pub lower: Tensor<R>,
    /// Upper bounds for every function value throughout each coordinate interval.
    pub upper: Tensor<R>,
}

/// Integral values and certified bounds, conditional on the supplied mathematical enclosures.
///
/// Every tensor matches the endpoint shape, dtype, and configured device.
/// Zero-width lanes contain exact zeros. Empty batches retain empty tensors.
#[derive(Debug, Clone)]
pub struct CertifiedSimpsonResult<R: Runtime<DType = DType>> {
    pub(crate) value: Tensor<R>,
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
    pub(crate) abs_error_bound: Tensor<R>,
    pub(crate) quadrature_error_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> CertifiedSimpsonResult<R> {
    /// Nominal composite Simpson integral for each original batch lane.
    pub fn value(&self) -> &Tensor<R> {
        &self.value
    }

    /// Lower endpoints enclosing the exact integrals under the callback guarantees.
    pub fn lower(&self) -> &Tensor<R> {
        &self.lower
    }

    /// Upper endpoints enclosing the exact integrals under the callback guarantees.
    pub fn upper(&self) -> &Tensor<R> {
        &self.upper
    }

    /// Nonnegative bounds on total nominal error, including arithmetic and quadrature uncertainty.
    pub fn abs_error_bound(&self) -> &Tensor<R> {
        &self.abs_error_bound
    }

    /// Nonnegative bounds on Simpson quadrature error alone.
    pub fn quadrature_error_bound(&self) -> &Tensor<R> {
        &self.quadrature_error_bound
    }
}
