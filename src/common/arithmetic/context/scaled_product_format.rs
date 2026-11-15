//! Exact format metadata for scaled positive products.

use crate::DType;
use crate::common::arithmetic::error::{ArithmeticError, ArithmeticResult};
use numr::error::Error;

pub(super) struct ProductRatioFormat {
    pub(super) steps: &'static [i64],
    pub(super) minimum_exponent: i64,
    pub(super) maximum_exponent: i64,
    pub(super) maximum: f64,
}

impl ProductRatioFormat {
    pub(super) fn for_dtype(dtype: DType) -> ArithmeticResult<Self> {
        match dtype {
            DType::F32 => Ok(Self {
                steps: &[64, 64, 32, 16, 8, 4, 2, 1],
                minimum_exponent: -149,
                maximum_exponent: 127,
                maximum: f64::from(f32::MAX),
            }),
            DType::F64 => Ok(Self {
                steps: &[512, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                minimum_exponent: -1074,
                maximum_exponent: 1023,
                maximum: f64::MAX,
            }),
            _ => Err(ArithmeticError::UnsupportedDType { dtype }),
        }
    }

    pub(super) fn check_factor_count(
        &self,
        numerator: usize,
        denominator: usize,
    ) -> ArithmeticResult<()> {
        let count = numerator
            .checked_add(denominator)
            .and_then(|value| i64::try_from(value).ok());
        // Every factor contributes at most this exponent magnitude plus two renormalization steps.
        let contribution = (-self.minimum_exponent).max(self.maximum_exponent) + 2;
        if count
            .and_then(|value| value.checked_mul(contribution))
            .is_none()
        {
            return Err(Error::InvalidArgument {
                arg: "positive_product_ratio",
                reason: format!("Factor counts {numerator}+{denominator} exceed safe I64 exponent accumulation. Split the product ratio."),
            }.into());
        }
        Ok(())
    }
}

/// The structural scaling schedule only requests normal F64 powers with exponents in -512..=512.
pub(super) fn scaling_power(exponent: i64) -> f64 {
    f64::from_bits(((1023 + exponent) as u64) << 52)
}
