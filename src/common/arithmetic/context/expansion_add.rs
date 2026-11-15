//! Add or subtract expansion centers in one compressing kernel.

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
    /// Add matching expansion centers before projecting their error radii.
    /// Binary operands require identical shapes, dtypes, and devices. No implicit broadcasting applies.
    pub fn expansion_add(
        &mut self,
        a: &CertifiedExpansion<R>,
        b: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.expansion_metadata(a)?;
        self.expansion_metadata(b)?;
        self.expansion_matching(&a.high, &b.high)?;
        let status = self.arithmetic.status();
        let (high, low, abs_error) =
            self.client()
                .expansion_add(a.operand(), b.operand(), status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }

    /// Subtract matching expansion centers before projecting their error radii.
    /// Binary operands require identical shapes, dtypes, and devices. No implicit broadcasting applies.
    pub fn expansion_sub(
        &mut self,
        a: &CertifiedExpansion<R>,
        b: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.expansion_metadata(a)?;
        self.expansion_metadata(b)?;
        self.expansion_matching(&a.high, &b.high)?;
        let status = self.arithmetic.status();
        let (high, low, abs_error) =
            self.client()
                .expansion_sub(a.operand(), b.operand(), status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }
}
