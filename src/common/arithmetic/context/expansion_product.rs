//! Multiply expansion centers and enclose discarded product residuals in one kernel.

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
    /// Enclose all four center products and both input radii with fused residuals.
    /// Components require matching shapes. The runtime must declare IEEE fused multiplication.
    /// Multiplication by exact one retains the other operand's component bits and full radius.
    pub fn expansion_mul(
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
                .expansion_mul(a.operand(), b.operand(), status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }
}
