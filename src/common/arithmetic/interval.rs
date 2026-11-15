//! Signed outward intervals using the caller's admitted basic arithmetic.

use super::error::{ArithmeticError, ArithmeticResult};
use crate::DType;
use numr::error::Error;
use numr::ops::{CompareOps, IntervalEnclosure, IntervalOperand, NextRepresentableOps, TensorOps};
use numr::runtime::{Device, Runtime, RuntimeClient, SquareRootContract};
use numr::tensor::Tensor;

/// Tensor endpoints enclosing real values elementwise.
///
/// Construct intervals through the checked arithmetic closure. Results remain provisional until that closure returns `Ok`.
/// Endpoint accessors expose storage aliases. Keep those tensors immutable and discard partial results after an error.
#[derive(Debug)]
pub struct Interval<R: Runtime<DType = DType>> {
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Interval<R> {
    /// Lower endpoints. Treat the returned tensor and its storage aliases as immutable.
    pub fn lower(&self) -> &Tensor<R> {
        &self.lower
    }

    /// Upper endpoints. Treat the returned tensor and its storage aliases as immutable.
    pub fn upper(&self) -> &Tensor<R> {
        &self.upper
    }
}

pub(crate) struct IntervalArithmetic<'a, R: Runtime<DType = DType>, C> {
    client: &'a C,
    dtype: DType,
    status: Tensor<R>,
}

/// Borrow interval endpoints as a kernel operand.
pub(crate) fn operand<R: Runtime<DType = DType>>(value: &Interval<R>) -> IntervalOperand<'_, R> {
    IntervalOperand {
        lower: &value.lower,
        upper: &value.upper,
    }
}

impl<'a, R, C> IntervalArithmetic<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    pub(crate) fn new(client: &'a C, dtype: DType) -> ArithmeticResult<Self> {
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(ArithmeticError::UnsupportedDType { dtype });
        }
        Ok(Self {
            client,
            dtype,
            status: client.interval_status()?,
        })
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn client(&self) -> &'a C {
        self.client
    }

    /// Device status flag shared by every operation of this calculation.
    pub(crate) fn status(&self) -> &Tensor<R> {
        &self.status
    }

    pub(crate) fn point(&mut self, value: &Tensor<R>) -> ArithmeticResult<Interval<R>> {
        self.bounds(value.clone(), value.clone())
    }

    pub(crate) fn bounds(
        &mut self,
        lower: Tensor<R>,
        upper: Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        let interval = Interval { lower, upper };
        self.metadata(&interval)?;
        self.client
            .interval_flag_bounds(operand(&interval), &self.status)?;
        Ok(interval)
    }

    /// Copy endpoints into fresh storage and flag invalid bounds in one pass.
    /// Equal views share one copy.
    pub(crate) fn capture(
        &mut self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.endpoint_metadata(lower, upper)?;
        let (lower, upper) = self
            .client
            .interval_capture(IntervalOperand { lower, upper }, &self.status)?;
        Ok(Interval { lower, upper })
    }

    fn binary(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
        enclosure: IntervalEnclosure,
        op: Binary,
    ) -> ArithmeticResult<Interval<R>> {
        self.metadata(a)?;
        self.metadata(b)?;
        let (a, b, status) = (operand(a), operand(b), &self.status);
        let (lower, upper) = match op {
            Binary::Add => self.client.interval_add(a, b, enclosure, status)?,
            Binary::Sub => self.client.interval_sub(a, b, enclosure, status)?,
            Binary::Mul => self.client.interval_mul(a, b, enclosure, status)?,
            Binary::DivPositiveInterval => self
                .client
                .interval_div_positive_interval(a, b, enclosure, status)?,
        };
        Ok(Interval { lower, upper })
    }

    /// Outward sum with exact-zero identities.
    pub(crate) fn add(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Outward, Binary::Add)
    }

    /// Outward sum that also retains TwoSum-exact singleton sums.
    pub(crate) fn add_tightened(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Tightened, Binary::Add)
    }

    /// Outward difference with exact-zero identities.
    pub(crate) fn sub(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Outward, Binary::Sub)
    }

    /// Outward difference that also retains TwoSum-exact singleton differences.
    pub(crate) fn sub_tightened(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Tightened, Binary::Sub)
    }

    /// Outward product over all signed endpoint products.
    pub(crate) fn mul(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Outward, Binary::Mul)
    }

    /// Outward product with singleton retention and proved-sign restriction.
    pub(crate) fn mul_tightened(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(a, b, IntervalEnclosure::Tightened, Binary::Mul)
    }

    fn divide(
        &mut self,
        a: &Interval<R>,
        exact_denominator: &Tensor<R>,
        enclosure: IntervalEnclosure,
    ) -> ArithmeticResult<Interval<R>> {
        self.metadata(a)?;
        self.tensor_metadata(exact_denominator)?;
        let (lower, upper) = self.client.interval_div_positive(
            operand(a),
            exact_denominator,
            enclosure,
            &self.status,
        )?;
        Ok(Interval { lower, upper })
    }

    /// Outward quotient by represented divisors that must be finite and strictly positive.
    pub(crate) fn div_positive(
        &mut self,
        a: &Interval<R>,
        exact_denominator: &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.divide(a, exact_denominator, IntervalEnclosure::Outward)
    }

    /// Positive division with singleton retention and proved-sign restriction.
    pub(crate) fn div_positive_tightened(
        &mut self,
        a: &Interval<R>,
        exact_denominator: &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.divide(a, exact_denominator, IntervalEnclosure::Tightened)
    }

    /// Outward quotient by intervals with strictly positive finite lower endpoints.
    #[cfg(test)]
    pub(crate) fn div_positive_interval(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(
            a,
            b,
            IntervalEnclosure::Outward,
            Binary::DivPositiveInterval,
        )
    }

    /// Positive interval division with singleton retention and proved-sign restriction.
    pub(crate) fn div_positive_interval_tightened(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.binary(
            a,
            b,
            IntervalEnclosure::Tightened,
            Binary::DivPositiveInterval,
        )
    }

    pub(crate) fn sqrt_nonnegative(&mut self, a: &Interval<R>) -> ArithmeticResult<Interval<R>> {
        if R::square_root_contract(self.dtype) != SquareRootContract::Ieee754RoundToNearest {
            return Err(ArithmeticError::UnsupportedArithmetic {
                dtype: self.dtype,
                failed_probes: vec!["square_root_contract"],
            });
        }
        self.metadata(a)?;
        let (lower, upper) = self
            .client
            .interval_sqrt_nonnegative(operand(a), &self.status)?;
        Ok(Interval { lower, upper })
    }

    pub(crate) fn mark_invalid(&mut self, mask: &Tensor<R>) -> ArithmeticResult<()> {
        if !mask.device().is_same(self.client.device()) {
            return Err(Error::DeviceMismatch.into());
        }
        if mask.numel() == 0 {
            return Ok(());
        }
        Ok(self.client.interval_flag_nonzero(mask, &self.status)?)
    }

    pub(crate) fn finite(&mut self, value: &Tensor<R>) -> ArithmeticResult<()> {
        self.tensor_metadata(value)?;
        Ok(self.client.interval_flag_nonfinite(value, &self.status)?)
    }

    /// Check accumulated device status once, after a complete arithmetic stage.
    pub(crate) fn check(&mut self, stage: &'static str) -> ArithmeticResult<()> {
        if self.status.item::<u32>()? != 0 {
            return Err(ArithmeticError::Range { stage });
        }
        Ok(())
    }

    pub(crate) fn tensor_metadata(&self, value: &Tensor<R>) -> ArithmeticResult<()> {
        if value.dtype() != self.dtype {
            return Err(Error::InvalidArgument {
                arg: "interval",
                reason: format!(
                    "Interval dtype {:?} differs from {:?}. Use matching interval dtypes.",
                    value.dtype(),
                    self.dtype
                ),
            }
            .into());
        }
        if !value.device().is_same(self.client.device()) {
            return Err(Error::DeviceMismatch.into());
        }
        Ok(())
    }

    pub(crate) fn metadata(&self, interval: &Interval<R>) -> ArithmeticResult<()> {
        self.endpoint_metadata(&interval.lower, &interval.upper)
    }

    pub(crate) fn endpoint_metadata(
        &self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
    ) -> ArithmeticResult<()> {
        let interval = IntervalOperand { lower, upper };
        self.tensor_metadata(interval.lower)?;
        self.tensor_metadata(interval.upper)?;
        if interval.lower.shape() != interval.upper.shape() {
            return Err(Error::InvalidArgument {
                arg: "interval",
                reason: format!(
                    "Interval shapes {:?} and {:?} differ. Use matching endpoint shapes.",
                    interval.lower.shape(),
                    interval.upper.shape()
                ),
            }
            .into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Sub,
    Mul,
    DivPositiveInterval,
}

#[cfg(test)]
#[path = "interval_tests.rs"]
mod tests;
