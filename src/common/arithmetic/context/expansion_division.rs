//! Correct a quotient center and certify it through an independent residual enclosure, in one kernel.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use crate::common::arithmetic::expansion::CertifiedExpansion;
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
    /// Divide by an expansion whose interval lower endpoint proves strict positivity.
    /// Certify the corrected center through its residual, including both input radii.
    /// Components require matching shapes. The runtime must declare IEEE fused multiplication.
    pub fn expansion_div_positive(
        &mut self,
        a: &CertifiedExpansion<R>,
        b: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.expansion_require_fma()?;
        self.expansion_metadata(a)?;
        self.expansion_metadata(b)?;
        self.expansion_matching(&a.high, &b.high)?;
        let status = self.arithmetic.status();
        let (high, low, abs_error) =
            self.client()
                .expansion_div_positive(a.operand(), b.operand(), status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }
}
