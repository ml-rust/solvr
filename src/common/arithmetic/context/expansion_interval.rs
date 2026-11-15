//! Explicitly project expansion centers into ordinary outward intervals.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use crate::common::arithmetic::expansion::CertifiedExpansion;
use crate::common::arithmetic::interval::Interval;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<'a, R, C> IntervalContext<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Project an expansion into an ordinary outward interval.
    /// This lossy projection rounds the center sum. Apply it after cancellation-sensitive expansion operations.
    pub fn expansion_interval(
        &mut self,
        value: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.expansion_metadata(value)?;
        let status = self.arithmetic.status();
        let (lower, upper) = self.client().expansion_interval(value.operand(), status)?;
        Ok(Interval { lower, upper })
    }
}
