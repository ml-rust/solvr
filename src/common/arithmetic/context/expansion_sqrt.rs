//! Qualified square-root certificates with complete exact-square proofs.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use crate::common::arithmetic::expansion::CertifiedExpansion;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
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
    /// Enclose the square root of a complete nonnegative expansion certificate.
    ///
    /// Qualified interval square roots provide the full-uncertainty fallback.
    /// Exact zero-radius squares retain a point only when the complete scaled residual proves equality.
    /// Scaling underflow can leave that proof inconclusive. Its arithmetic radii remain in the proof.
    /// Input projection and proof arithmetic retain their existing qualification and finite-range requirements.
    pub fn expansion_sqrt_nonnegative(
        &mut self,
        value: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        let interval = self.expansion_interval(value)?;
        let root = self.sqrt_nonnegative(&interval)?;
        let fallback = self.expansion_from_interval(&root)?;
        let c = self.client();
        let zero = Tensor::zeros(value.high().shape(), value.high().dtype(), c.device())?;
        let one = Tensor::ones(value.high().shape(), value.high().dtype(), c.device())?;
        let point_input = c.eq(value.abs_error(), &zero)?;

        // Nonpoint rows retain the fallback and use harmless operands in the optional exactness proof.
        let candidate =
            self.expansion_point(&c.where_cond(&point_input, fallback.high(), &one)?)?;
        let proof_input = self.expansion_bounds(
            &c.where_cond(&point_input, value.high(), &one)?,
            &c.where_cond(&point_input, value.low(), &zero)?,
            &c.where_cond(&point_input, value.abs_error(), &zero)?,
        )?;
        let half = self.expansion_point(&c.mul_scalar(&one, 0.5)?)?;
        let quarter = self.expansion_point(&c.mul_scalar(&one, 0.25)?)?;
        let half_candidate = self.expansion_mul(&candidate, &half)?;
        let square = self.expansion_mul(&half_candidate, &half_candidate)?;
        let quarter_input = self.expansion_mul(&proof_input, &quarter)?;
        let residual = self.expansion_sub(&square, &quarter_input)?;
        let exact_center = c.mul(
            &c.eq(residual.high(), &zero)?,
            &c.eq(residual.low(), &zero)?,
        )?;
        let exact = c.mul(
            &point_input,
            &c.mul(&exact_center, &c.eq(residual.abs_error(), &zero)?)?,
        )?;
        self.expansion_bounds(
            fallback.high(),
            &c.where_cond(&exact, &zero, fallback.low())?,
            &c.where_cond(&exact, &zero, fallback.abs_error())?,
        )
    }
}
