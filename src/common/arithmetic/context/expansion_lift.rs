//! Preserve complete interval uncertainty in an owned expansion.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use crate::common::arithmetic::expansion::CertifiedExpansion;
use crate::common::arithmetic::interval::{Interval, operand};
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
    /// Lift a complete interval into an owned expansion center and radius.
    ///
    /// The radius retains the whole interval. Equal endpoints preserve the lower endpoint's bits, including signed zero.
    /// Combine separately retained small terms before projection. Lifting a rounded interval cannot recover lost terms.
    /// Finite components do not guarantee a finite later projection near dtype endpoints.
    pub fn expansion_from_interval(
        &mut self,
        value: &Interval<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.arithmetic.metadata(value)?;
        let status = self.arithmetic.status();
        let (high, low, abs_error) = self
            .client()
            .expansion_from_interval(operand(value), status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }
}
