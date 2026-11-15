//! Normalize positive represented factors using exact masked binary scaling.

use super::{
    IntervalContext,
    scaled_product_format::{ProductRatioFormat, scaling_power},
};
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use numr::{
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct ScaledPositive<R: Runtime<DType = DType>> {
    pub(super) mantissa: Tensor<R>,
    pub(super) exponent: Tensor<R>,
}

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
    pub(super) fn normalize_positive_factor(
        &self,
        value: &Tensor<R>,
        format: &ProductRatioFormat,
    ) -> ArithmeticResult<ScaledPositive<R>> {
        let mut result = ScaledPositive {
            mantissa: self.capture(value)?,
            exponent: Tensor::zeros(value.shape(), DType::I64, self.client().device())?,
        };
        let one = Tensor::ones(&[], self.dtype(), self.client().device())?;
        for &step in format.steps {
            let large = Tensor::full_scalar(
                &[],
                self.dtype(),
                scaling_power(step),
                self.client().device(),
            )?;
            let small = Tensor::full_scalar(
                &[],
                self.dtype(),
                scaling_power(-step),
                self.client().device(),
            )?;
            let down = self.client().ge(&result.mantissa, &large)?;
            self.scale_positive(&mut result, &down, &small, step)?;
            let up = self.client().lt(&result.mantissa, &small)?;
            self.scale_positive(&mut result, &up, &large, -step)?;
        }
        let up = self.client().lt(&result.mantissa, &one)?;
        let two = Tensor::full_scalar(&[], self.dtype(), 2.0, self.client().device())?;
        self.scale_positive(&mut result, &up, &two, -1)?;
        Ok(result)
    }

    /// Qualified normalized products and quotients have lower endpoints between one quarter and four.
    pub(super) fn renormalize_positive(
        &self,
        value: &mut ScaledPositive<R>,
    ) -> ArithmeticResult<()> {
        let c = self.client();
        let one = Tensor::ones(&[], self.dtype(), c.device())?;
        let two = Tensor::full_scalar(&[], self.dtype(), 2.0, c.device())?;
        let half = Tensor::full_scalar(&[], self.dtype(), 0.5, c.device())?;
        let down = c.ge(&value.mantissa, &two)?;
        self.scale_positive(value, &down, &half, 1)?;
        for _ in 0..2 {
            let up = c.lt(&value.mantissa, &one)?;
            self.scale_positive(value, &up, &two, -1)?;
        }
        Ok(())
    }

    fn scale_positive(
        &self,
        value: &mut ScaledPositive<R>,
        mask: &Tensor<R>,
        factor: &Tensor<R>,
        exponent: i64,
    ) -> ArithmeticResult<()> {
        let c = self.client();
        let one = Tensor::ones(&[], self.dtype(), c.device())?;
        // Mask both operands before eager multiplication, including unselected large powers.
        let operand = c.where_cond(mask, &value.mantissa, &one)?;
        let factor = c.where_cond(mask, factor, &one)?;
        let scaled = c.mul(&operand, &factor)?;
        value.mantissa = c.where_cond(mask, &scaled, &value.mantissa)?;
        let zero = Tensor::zeros(&[], DType::I64, c.device())?;
        let delta = Tensor::full_scalar(&[], DType::I64, exponent as f64, c.device())?;
        let delta = c.where_cond(mask, &delta, &zero)?;
        value.exponent = c.add(&value.exponent, &delta)?;
        Ok(())
    }
}
