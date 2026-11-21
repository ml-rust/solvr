//! Interval arc cosine and logarithm over correctly rounded F64 kernels.

use super::IntervalContext;
use crate::DType;
use crate::common::arithmetic::error::{ArithmeticError, ArithmeticResult};
use crate::common::arithmetic::interval::Interval;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{FusedMulAddContract, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Largest F64 below pi.
const PI_LOWER: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
/// Smallest F64 above pi.
const PI_UPPER: f64 = f64::from_bits(0x4009_21fb_5444_2d19);

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
    /// Enclose the arc cosine of intervals inside `[-1, 1]`, in radians.
    ///
    /// The result is `[next_down(acos(upper)), next_up(acos(lower))]` with numr's `cr_acos`.
    /// The lower endpoint clamps to 0 and the upper endpoint to the F64 above pi.
    /// An input endpoint of exactly 1 maps to 0. An input endpoint of exactly -1 maps to the F64 pi pair.
    /// Endpoints outside `[-1, 1]` reject the enclosing call at its completion check.
    /// Those lanes evaluate the nearest of -1 and 1 instead.
    ///
    /// The context dtype must be F64, and the runtime must declare IEEE fused multiply-add for F64.
    /// Otherwise the call returns `UnsupportedArithmetic`, including for empty inputs.
    pub fn acos(&mut self, a: &Interval<R>) -> ArithmeticResult<Interval<R>>
    where
        C: CorrectlyRoundedOps<R>,
    {
        self.require_correctly_rounded()?;
        self.arithmetic.metadata(a)?;
        let c = self.client();
        let constant =
            |value: f64| Tensor::<R>::full_scalar(a.lower().shape(), DType::F64, value, c.device());
        let (minus_one, one, zero) = (constant(-1.0)?, constant(1.0)?, constant(0.0)?);

        // Flag each endpoint outside [-1, 1], then evaluate it at the nearest domain edge.
        let mut inside = Vec::with_capacity(2);
        for endpoint in [a.lower(), a.upper()] {
            let below = c.lt(endpoint, &minus_one)?;
            let above = c.gt(endpoint, &one)?;
            self.arithmetic.mark_invalid(&below)?;
            self.arithmetic.mark_invalid(&above)?;
            let raised = c.where_cond(&below, &minus_one, endpoint)?;
            inside.push(c.where_cond(&above, &one, &raised)?);
        }
        let (lower_in, upper_in) = (&inside[0], &inside[1]);

        // acos decreases, so the upper input endpoint gives the lower result endpoint.
        let lower = c.next_down(&c.cr_acos(upper_in)?)?;
        let lower = c.where_cond(&c.lt(&lower, &zero)?, &zero, &lower)?;
        let lower = c.where_cond(&c.eq(upper_in, &minus_one)?, &constant(PI_LOWER)?, &lower)?;

        let pi_upper = constant(PI_UPPER)?;
        let upper = c.next_up(&c.cr_acos(lower_in)?)?;
        let upper = c.where_cond(&c.gt(&upper, &pi_upper)?, &pi_upper, &upper)?;
        let upper = c.where_cond(&c.eq(lower_in, &one)?, &zero, &upper)?;
        self.arithmetic.bounds(lower, upper)
    }

    /// Enclose the natural logarithm of intervals with strictly positive lower endpoints.
    ///
    /// The result is `[next_down(log(lower)), next_up(log(upper))]` with numr's `cr_log`.
    /// A lower endpoint `<= 0` rejects the enclosing call at its completion check.
    /// Nonpositive endpoints evaluate 1 instead. An infinite upper endpoint also rejects.
    ///
    /// The context dtype must be F64, and the runtime must declare IEEE fused multiply-add for F64.
    /// Otherwise the call returns `UnsupportedArithmetic`, including for empty inputs.
    pub fn log_positive(&mut self, a: &Interval<R>) -> ArithmeticResult<Interval<R>>
    where
        C: CorrectlyRoundedOps<R>,
    {
        self.require_correctly_rounded()?;
        self.arithmetic.metadata(a)?;
        let c = self.client();
        let shape = a.lower().shape();
        let zero = Tensor::<R>::zeros(shape, DType::F64, c.device())?;
        let one = Tensor::<R>::ones(shape, DType::F64, c.device())?;

        let lower_invalid = c.le(a.lower(), &zero)?;
        self.arithmetic.mark_invalid(&lower_invalid)?;
        let lower_in = c.where_cond(&lower_invalid, &one, a.lower())?;
        let upper_in = c.where_cond(&c.le(a.upper(), &zero)?, &one, a.upper())?;

        // log increases, so each endpoint pads outward on its own side.
        let lower = c.next_down(&c.cr_log(&lower_in)?)?;
        let upper = c.next_up(&c.cr_log(&upper_in)?)?;
        self.arithmetic.bounds(lower, upper)
    }

    /// Admit F64 contexts on runtimes that declare the correctly rounded kernel contracts.
    fn require_correctly_rounded(&self) -> ArithmeticResult<()> {
        let dtype = self.dtype();
        if dtype != DType::F64
            || R::fused_mul_add_contract(dtype) != FusedMulAddContract::Ieee754Fused
        {
            return Err(ArithmeticError::UnsupportedArithmetic {
                dtype,
                failed_probes: vec!["correctly_rounded_contract"],
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn pi_constants_bracket_pi() {
        assert_eq!(PI_LOWER, std::f64::consts::PI);
        assert_eq!(PI_UPPER.to_bits(), PI_LOWER.to_bits() + 1);
        // pi = 0x1.921fb54442d18469898cc...p+1, so the F64 nearest pi lies below it.
        const { assert!(PI_LOWER < PI_UPPER) };
    }

    #[test]
    fn f32_contexts_reject_correctly_rounded_operations() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let result = crate::common::arithmetic::with_interval_arithmetic::<CpuRuntime, _, _, _>(
            &client,
            DType::F32,
            "f32 acos",
            |context| {
                let input = Tensor::from_slice(&[0.5f32], &[1], &device)?;
                let point = context.point(&input)?;
                context.acos(&point)
            },
        );
        assert!(matches!(
            result,
            Err(ArithmeticError::UnsupportedArithmetic { dtype: DType::F32, ref failed_probes })
                if failed_probes == &["correctly_rounded_contract"]
        ));
    }
}
