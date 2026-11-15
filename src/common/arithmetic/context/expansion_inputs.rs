//! Snapshot and check expansion components without host status reads.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::{ArithmeticError, ArithmeticResult};
use crate::common::arithmetic::expansion::CertifiedExpansion;
use numr::error::Error;
use numr::ops::{
    CompareOps, ExpansionOperand, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{FusedMulAddContract, Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
    /// Snapshot represented values into exact expansion centers with zero radii.
    /// Results remain provisional until the enclosing arithmetic call returns `Ok`.
    pub fn expansion_point(
        &mut self,
        value: &Tensor<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.check_capture(value)?;
        let status = self.arithmetic.status();
        let (high, low, abs_error) = self.client().expansion_point(value, status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }

    /// Snapshot matching components and normalize their exact center sum with TwoSum.
    /// Zero low components preserve both original center bit patterns.
    /// Nonfinite components or negative radii reject the enclosing call at completion.
    /// Components require identical shapes, dtypes, and devices. No implicit broadcasting applies.
    pub fn expansion_bounds(
        &mut self,
        high: &Tensor<R>,
        low: &Tensor<R>,
        abs_error: &Tensor<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        self.expansion_matching(high, low)?;
        self.expansion_matching(high, abs_error)?;
        for component in [high, low, abs_error] {
            self.check_capture(component)?;
        }
        let value = ExpansionOperand {
            high,
            low,
            abs_error,
        };
        let status = self.arithmetic.status();
        let (high, low, abs_error) = self.client().expansion_normalize(value, status)?;
        Ok(CertifiedExpansion {
            high,
            low,
            abs_error,
        })
    }

    pub(super) fn expansion_metadata(&self, value: &CertifiedExpansion<R>) -> ArithmeticResult<()> {
        self.expansion_matching(&value.high, &value.low)?;
        self.expansion_matching(&value.high, &value.abs_error)
    }

    pub(super) fn expansion_require_fma(&self) -> ArithmeticResult<()> {
        let dtype = self.arithmetic.dtype();
        if R::fused_mul_add_contract(dtype) != FusedMulAddContract::Ieee754Fused {
            return Err(ArithmeticError::UnsupportedArithmetic {
                dtype,
                failed_probes: vec!["fused_mul_add_contract"],
            });
        }
        Ok(())
    }

    pub(super) fn expansion_matching(&self, a: &Tensor<R>, b: &Tensor<R>) -> ArithmeticResult<()> {
        self.arithmetic.tensor_metadata(a)?;
        self.arithmetic.tensor_metadata(b)?;
        if a.shape() != b.shape() {
            return Err(Error::InvalidArgument {
                arg: "expansion",
                reason: format!(
                    "Expansion shapes {:?} and {:?} differ. Use matching shapes without implicit broadcasting.",
                    a.shape(), b.shape()
                ),
            }.into());
        }
        Ok(())
    }
}
