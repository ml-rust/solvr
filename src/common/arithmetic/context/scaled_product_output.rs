//! Construct exact represented dyadics without stepping below the smallest subnormal.

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
    pub(super) fn positive_dyadic_lower(
        &mut self,
        exponent: &Tensor<R>,
        format: &ProductRatioFormat,
    ) -> ArithmeticResult<Tensor<R>> {
        let c = self.client();
        let minimum =
            Tensor::full_scalar(&[], DType::I64, format.minimum_exponent as f64, c.device())?;
        let maximum =
            Tensor::full_scalar(&[], DType::I64, format.maximum_exponent as f64, c.device())?;
        let below = c.lt(exponent, &minimum)?;
        self.arithmetic.mark_invalid(&below)?;
        self.arithmetic
            .check("positive product ratio lower bound lacks represented resolution")?;
        let above = c.gt(exponent, &maximum)?;
        let zero = Tensor::zeros(&[], DType::I64, c.device())?;
        let mut remaining = c.where_cond(&above, &zero, exponent)?;
        let one = Tensor::ones(&[], self.dtype(), c.device())?;
        let mut result = Tensor::ones(exponent.shape(), self.dtype(), c.device())?;
        for &step in format.steps {
            let positive = Tensor::full_scalar(&[], DType::I64, step as f64, c.device())?;
            let negative = Tensor::full_scalar(&[], DType::I64, -step as f64, c.device())?;
            for (mask, power, delta) in [
                (c.ge(&remaining, &positive)?, scaling_power(step), &positive),
                (
                    c.le(&remaining, &negative)?,
                    scaling_power(-step),
                    &negative,
                ),
            ] {
                let factor = Tensor::full_scalar(&[], self.dtype(), power, c.device())?;
                let operand = c.where_cond(&mask, &result, &one)?;
                let factor = c.where_cond(&mask, &factor, &one)?;
                let product = c.mul(&operand, &factor)?;
                result = c.where_cond(&mask, &product, &result)?;
                remaining = c.sub(&remaining, &c.where_cond(&mask, delta, &zero)?)?;
            }
        }
        let largest = Tensor::full_scalar(&[], self.dtype(), format.maximum, c.device())?;
        let result = c.where_cond(&above, &largest, &result)?;
        self.arithmetic.finite(&result)?;
        Ok(result)
    }
}
