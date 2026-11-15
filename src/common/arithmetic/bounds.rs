//! Dtype-specific positive uncertainty bounds with exact zero identities.

use super::error::{ArithmeticError, ArithmeticResult};
use super::precision::Precision;
use crate::DType;
use numr::ops::{CompareOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) struct Bounds<'a, R: Runtime<DType = DType>, C> {
    client: &'a C,
    zero: Tensor<R>,
    epsilon: Tensor<R>,
    eta: Tensor<R>,
    invalid: Tensor<R>,
}
impl<'a, R, C> Bounds<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    pub(crate) fn new(client: &'a C, query: &Tensor<R>) -> ArithmeticResult<Self> {
        let precision = Precision::for_dtype(query.dtype())?;
        let constant =
            |value| Tensor::full_scalar(query.shape(), query.dtype(), value, client.device());
        let zero = constant(0.0)?;
        Ok(Self {
            client,
            invalid: zero.clone(),
            zero,
            epsilon: constant(precision.epsilon)?,
            eta: constant(precision.minimum_subnormal)?,
        })
    }

    pub(crate) fn mark(&mut self, mask: &Tensor<R>) -> ArithmeticResult<()> {
        self.invalid = self.client.maximum(&self.invalid, mask)?;
        Ok(())
    }

    pub(crate) fn finite(&mut self, value: &Tensor<R>) -> ArithmeticResult<()> {
        let invalid = self.client.maximum(
            &self
                .client
                .cast(&self.client.isnan(value)?, value.dtype())?,
            &self
                .client
                .cast(&self.client.isinf(value)?, value.dtype())?,
        )?;
        self.mark(&invalid)
    }

    pub(crate) fn rounding(&mut self, value: &Tensor<R>) -> ArithmeticResult<Tensor<R>> {
        let product = self.client.mul(&self.epsilon, &self.client.abs(value)?)?;
        let bound = self.client.add(&product, &self.eta)?;
        self.finite(&bound)?;
        Ok(bound)
    }

    pub(crate) fn add(&mut self, a: &Tensor<R>, b: &Tensor<R>) -> ArithmeticResult<Tensor<R>> {
        let a_zero = self.client.eq(a, &self.zero)?;
        let b_zero = self.client.eq(b, &self.zero)?;
        let both = self.client.mul(
            &self.client.gt(a, &self.zero)?,
            &self.client.gt(b, &self.zero)?,
        )?;
        let sum = self.client.add(a, b)?;
        self.finite(&sum)?;
        let used = self.client.where_cond(&both, &sum, &self.zero)?;
        let padding = self.rounding(&used)?;
        let padded = self.client.add(&used, &padding)?;
        let bound =
            self.client
                .where_cond(&a_zero, b, &self.client.where_cond(&b_zero, a, &padded)?)?;
        self.finite(&bound)?;
        Ok(bound)
    }

    pub(crate) fn multiply(&mut self, a: &Tensor<R>, b: &Tensor<R>) -> ArithmeticResult<Tensor<R>> {
        let positive = self.client.mul(
            &self.client.gt(a, &self.zero)?,
            &self.client.gt(b, &self.zero)?,
        )?;
        let product = self.client.mul(a, b)?;
        self.finite(&product)?;
        let padding = self.rounding(&product)?;
        let padded = self.client.add(&product, &padding)?;
        let bound = self.client.where_cond(&positive, &padded, &self.zero)?;
        self.finite(&bound)?;
        Ok(bound)
    }

    pub(crate) fn divide(
        &mut self,
        numerator: &Tensor<R>,
        denominator: &Tensor<R>,
    ) -> ArithmeticResult<Tensor<R>> {
        let quotient = self.client.div(numerator, denominator)?;
        self.finite(&quotient)?;
        let padding = self.rounding(&quotient)?;
        let padded = self.client.add(&quotient, &padding)?;
        let bound =
            self.client
                .where_cond(&self.client.eq(numerator, &self.zero)?, &self.zero, &padded)?;
        self.finite(&bound)?;
        Ok(bound)
    }

    pub(crate) fn finish(self) -> ArithmeticResult<()> {
        let value = self.client.any(&self.invalid, &[], false)?;
        let invalid = match self.invalid.dtype() {
            DType::F32 => value.item::<f32>()? != 0.0,
            DType::F64 => value.item::<f64>()? != 0.0,
            _ => return Err(ArithmeticError::ControlMaskDType),
        };
        if invalid {
            return Err(ArithmeticError::InvalidBounds);
        }
        Ok(())
    }
}
