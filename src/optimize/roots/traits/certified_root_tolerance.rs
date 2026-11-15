//! Shared inward-rounded root-width tolerance.

use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::OptimizeResult;
use numr::ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Absolute and relative width allowances in the caller's convergence coordinate.
#[derive(Clone, Copy, Debug)]
pub struct CertifiedRootTolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl CertifiedRootTolerance {
    /// Check complete bounds against the same inward tolerance used by certified root solvers.
    ///
    /// Inputs are finite ordered rank-one bounds matching the context's numerical dtype and device.
    /// The result is a matching numerical mask for `width <= atol + rtol * max(abs(lower), abs(upper))`.
    /// Width rounds outward. Tolerance arithmetic rounds inward, including F32 scalar conversion.
    /// Exact-width rows pass, including when both tolerances are zero.
    /// Use the caller's qualified context. This method does not complete or replace that context.
    /// The caller completes its existing arithmetic scope before accepting this result.
    pub fn satisfied<R, C>(
        &self,
        ctx: &mut IntervalContext<'_, R, C>,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
    ) -> OptimizeResult<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: TensorOps<R>
            + CompareOps<R>
            + MidpointOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        crate::optimize::roots::impl_generic::certified_bisection::convergence::admitted_satisfied(
            ctx, lower, upper, self,
        )
    }
}
