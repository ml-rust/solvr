//! Complete interval calculations under checked runtime arithmetic.

use super::context::IntervalContext;
use super::envelope;
use super::error::{ArithmeticError, ArithmeticResult};
use super::precision::Precision;
use crate::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{FloatArithmeticContract, Runtime, RuntimeClient};

/// Build interval expressions on the configured client and check them before returning results.
///
/// Select F32 or F64, then construct inputs and operations inside `build`.
/// `stage` identifies the calculation when its final bounds reject.
///
/// # Arithmetic admission
///
/// The runtime must declare IEEE basic arithmetic for the selected dtype.
/// Existing nearest-even, gradual-underflow, and TwoSum probes run before `build`.
/// Every executing thread must retain the same admitted arithmetic mode throughout the call.
/// Sampled probes do not establish every worker thread's mode.
/// Square-root operations independently require Numr's correctly rounded square-root contract.
///
/// # Completion and ownership
///
/// Closure results remain provisional until this function returns `Ok`.
/// Nonfinite, inverted, or invalid-domain bounds reject at the mandatory completion check.
/// Closure errors return unchanged. Interval results require the context status check before returning `Ok`.
/// Completion covers context operations. Check outputs from arbitrary client operations separately.
///
/// Closure side effects fall outside this guarantee. Discard captured partial results after `Err`.
/// Endpoint accessors permit storage aliases. Preserve endpoint immutability throughout their use.
/// Public interval constructors copy external storage through the configured client.
pub fn with_interval_arithmetic<'a, R, C, F, T>(
    client: &'a C,
    dtype: DType,
    stage: &'static str,
    build: F,
) -> ArithmeticResult<T>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: FnOnce(&mut IntervalContext<'a, R, C>) -> ArithmeticResult<T>,
{
    Precision::for_dtype(dtype)?;
    if R::float_arithmetic_contract(dtype) != FloatArithmeticContract::Ieee754Basic {
        return Err(ArithmeticError::UnsupportedArithmetic {
            dtype,
            failed_probes: vec!["provider_contract"],
        });
    }
    envelope::check(client, dtype)?;
    let mut context = IntervalContext::new(client, dtype)?;
    let result = build(&mut context)?;
    context.complete(stage)?;
    Ok(result)
}
