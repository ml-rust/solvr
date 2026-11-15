//! Same-device rounding, gradual-underflow, and TwoSum admission.

use super::error::{ArithmeticError, ArithmeticResult};
use super::precision::Precision;
use crate::DType;
use numr::ops::{CompareOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Probe names in numr's `interval_envelope` bit order.
const PROBES: [&str; 8] = [
    "nearest_even_positive_ties",
    "nearest_even_negative_ties",
    "gradual_subnormal_inputs",
    "gradual_multiplication_underflow",
    "gradual_division_underflow",
    "two_sum_positive_witness",
    "two_sum_negative_witness",
    "two_sum_cancellation_witness",
];

pub(crate) fn check<R, C>(client: &C, dtype: DType) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + TwoSumOps<R> + RuntimeClient<R>,
{
    Precision::for_dtype(dtype)?;
    // One device pass evaluates every probe in the arithmetic the interval kernels run.
    let bits = client.interval_envelope(dtype)?.item::<u32>()?;
    if bits > 255 {
        return Err(ArithmeticError::InvalidProbeControl {
            control: f64::from(bits),
        });
    }
    if bits != 0 {
        let failed_probes = PROBES
            .iter()
            .enumerate()
            .filter_map(|(index, name)| ((bits & (1u32 << index)) != 0).then_some(*name))
            .collect();
        return Err(ArithmeticError::UnsupportedArithmetic {
            dtype,
            failed_probes,
        });
    }
    Ok(())
}
