//! Exact host metadata for supported arithmetic dtypes.

use super::error::{ArithmeticError, ArithmeticResult};
use crate::DType;

pub(crate) struct Precision {
    pub epsilon: f64,
    pub minimum_subnormal: f64,
    pub minimum_normal: f64,
}
impl Precision {
    pub(crate) fn for_dtype(dtype: DType) -> ArithmeticResult<Self> {
        match dtype {
            DType::F32 => Ok(Self {
                epsilon: f64::from(f32::EPSILON),
                minimum_subnormal: f64::from(f32::from_bits(1)),
                minimum_normal: f64::from(f32::MIN_POSITIVE),
            }),
            DType::F64 => Ok(Self {
                epsilon: f64::EPSILON,
                minimum_subnormal: f64::from_bits(1),
                minimum_normal: f64::MIN_POSITIVE,
            }),
            _ => Err(ArithmeticError::UnsupportedDType { dtype }),
        }
    }
}
