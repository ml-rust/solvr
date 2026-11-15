//! Conservative positive dyadic bounds for product ratios with large intermediate exponents.

use super::{
    IntervalContext, scaled_product_format::ProductRatioFormat,
    scaled_product_normalize::ScaledPositive,
};
use crate::DType;
use crate::common::arithmetic::error::ArithmeticResult;
use numr::{
    error::Error,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::{Layout, Tensor},
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
    /// Return an owned finite positive lower bound for the ratio of positive represented factors.
    ///
    /// All factors require the context dtype/device and a joint broadcast shape. Empty lists return scalar one.
    /// Supplied empty tensors preserve their joint broadcast shape.
    /// The result is an exact represented power of two, except finite-maximum saturation.
    /// Finite positive output drops the accumulated mantissa and has no relative-accuracy guarantee.
    /// Exponents below the smallest subnormal return `Range`: the lower certificate lacks represented resolution.
    /// This error does not assert that the exact ratio underflows.
    pub fn positive_product_ratio_lower(
        &mut self,
        numerators: &[&Tensor<R>],
        denominators: &[&Tensor<R>],
    ) -> ArithmeticResult<Tensor<R>> {
        let format = ProductRatioFormat::for_dtype(self.dtype())?;
        format.check_factor_count(numerators.len(), denominators.len())?;
        let shape = self.admit_positive_product_factors(numerators, denominators)?;
        let mut value = ScaledPositive {
            mantissa: Tensor::ones(&shape, self.dtype(), self.client().device())?,
            exponent: Tensor::zeros(&shape, DType::I64, self.client().device())?,
        };
        for (factors, divide) in [(numerators, false), (denominators, true)] {
            for &factor in factors {
                let factor = self.normalize_positive_factor(factor, &format)?;
                let current = self.point(&value.mantissa)?;
                let normalized = self.point(&factor.mantissa)?;
                let bound = if divide {
                    self.div_positive_interval(&current, &normalized)?
                } else {
                    self.mul(&current, &normalized)?
                };
                value.mantissa = bound.lower;
                value.exponent = if divide {
                    self.client().sub(&value.exponent, &factor.exponent)?
                } else {
                    self.client().add(&value.exponent, &factor.exponent)?
                };
                self.renormalize_positive(&mut value)?;
            }
        }
        self.positive_dyadic_lower(&value.exponent, &format)
    }

    fn admit_positive_product_factors(
        &mut self,
        numerators: &[&Tensor<R>],
        denominators: &[&Tensor<R>],
    ) -> ArithmeticResult<Vec<usize>> {
        let mut shape = Vec::new();
        for &factor in numerators.iter().chain(denominators) {
            self.arithmetic.tensor_metadata(factor)?;
            shape = Layout::broadcast_shapes(&shape, factor.shape()).ok_or_else(|| Error::InvalidArgument {
                arg: "positive_product_ratio",
                reason: format!("Factor shape {:?} does not broadcast with {shape:?}. Supply compatible factor shapes.", factor.shape()),
            })?;
        }
        let zero = Tensor::zeros(&[], self.dtype(), self.client().device())?;
        for &factor in numerators.iter().chain(denominators) {
            self.arithmetic.finite(factor)?;
            self.arithmetic
                .mark_invalid(&self.client().le(factor, &zero)?)?;
        }
        // Check original factors before broadcasting. Invalid scalars cannot disappear into empty outputs.
        self.arithmetic
            .check("positive product ratio requires finite strictly positive factors")?;
        let count = shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
        let count = count.filter(|&value| u32::try_from(value).is_ok()).ok_or_else(|| Error::InvalidArgument {
            arg: "positive_product_ratio",
            reason: format!("Broadcast shape {shape:?} exceeds bounded U32 element indexing. Split the product ratio."),
        })?;
        count
            .checked_mul(self.dtype().size_in_bytes())
            .ok_or_else(|| Error::InvalidArgument {
                arg: "positive_product_ratio",
                reason: format!(
                    "Broadcast shape {shape:?} overflows allocation bytes. Split the product ratio."
                ),
            })?;
        Ok(shape)
    }
}
