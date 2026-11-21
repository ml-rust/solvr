//! Build interval expressions within a checked arithmetic call.

use super::error::ArithmeticResult;
use super::interval::{Interval, IntervalArithmetic};
use super::precision::Precision;
use crate::DType;
use numr::error::Error;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Interval operations available inside `with_interval_arithmetic`.
///
/// Construct inputs with `point` or `bounds`. Both copy external storage on the configured client.
/// Operation results remain provisional until the enclosing call returns `Ok`.
/// Keep every interval endpoint immutable, including tensors reached through endpoint accessors.
/// Multiplication and positive division retain signs proved by interval endpoints.
///
/// Normal Numr broadcasting applies between intervals. Endpoints within one interval require matching shapes.
/// Native view limits still apply. Input captures support at most `u32::MAX` elements.
/// The context has no public constructor or completion bypass.
pub struct IntervalContext<'a, R: Runtime<DType = DType>, C> {
    arithmetic: IntervalArithmetic<'a, R, C>,
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
    pub(crate) fn new(client: &'a C, dtype: DType) -> ArithmeticResult<Self> {
        Ok(Self {
            arithmetic: IntervalArithmetic::new(client, dtype)?,
        })
    }

    /// Borrow the configured client without replacing its runtime or execution config.
    pub fn client(&self) -> &'a C {
        self.arithmetic.client()
    }

    /// Return the selected arithmetic dtype.
    pub fn dtype(&self) -> DType {
        self.arithmetic.dtype()
    }

    /// Return the selected dtype's smallest positive normal value.
    pub fn minimum_normal(&self) -> ArithmeticResult<f64> {
        Ok(Precision::for_dtype(self.dtype())?.minimum_normal)
    }

    /// Copy represented values into singleton intervals.
    pub fn point(&mut self, value: &Tensor<R>) -> ArithmeticResult<Interval<R>> {
        self.check_capture(value)?;
        self.arithmetic.capture(value, value)
    }

    /// Copy lower and upper endpoints into intervals with matching metadata.
    ///
    /// Nonfinite or inverted bounds reject the enclosing call at its completion check.
    pub fn bounds(
        &mut self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.endpoint_metadata(lower, upper)?;
        self.check_capture(lower)?;
        self.check_capture(upper)?;
        self.arithmetic.capture(lower, upper)
    }

    /// Enclose interval addition with outward rounding and exact-zero identities.
    /// Guarded singleton operands retain a singleton result when TwoSum proves exactness.
    pub fn add(&mut self, a: &Interval<R>, b: &Interval<R>) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.add_tightened(a, b)
    }

    /// Enclose interval subtraction with outward rounding and exact-zero identities.
    /// Guarded singleton operands retain a singleton result when TwoSum proves exactness.
    pub fn sub(&mut self, a: &Interval<R>, b: &Interval<R>) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.sub_tightened(a, b)
    }

    /// Enclose all signed endpoint products, including nonzero underflow.
    /// Admitted fused arithmetic retains guarded singleton products when its residual proves exactness.
    pub fn mul(&mut self, a: &Interval<R>, b: &Interval<R>) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.mul_tightened(a, b)
    }

    /// Divide by copied, represented divisors that must be finite and strictly positive.
    /// Admitted fused arithmetic retains guarded singleton quotients when its residual proves exactness.
    pub fn div_positive(
        &mut self,
        a: &Interval<R>,
        exact_positive_divisor: &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.metadata(a)?;
        self.check_capture(exact_positive_divisor)?;
        self.arithmetic
            .div_positive_tightened(a, exact_positive_divisor)
    }

    /// Divide by intervals whose finite lower endpoints must be strictly positive.
    /// Admitted fused arithmetic retains guarded singleton quotients when its residual proves exactness.
    pub fn div_positive_interval(
        &mut self,
        a: &Interval<R>,
        b: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.div_positive_interval_tightened(a, b)
    }

    /// Enclose square roots of finite nonnegative intervals.
    ///
    /// The runtime must independently declare correctly rounded square roots for the selected dtype.
    /// Unsupported square-root providers return `UnsupportedArithmetic`, including empty inputs.
    pub fn sqrt_nonnegative(&mut self, a: &Interval<R>) -> ArithmeticResult<Interval<R>> {
        self.arithmetic.sqrt_nonnegative(a)
    }

    pub(crate) fn complete(&mut self, stage: &'static str) -> ArithmeticResult<()> {
        self.arithmetic.check(stage)
    }

    fn capture(&self, value: &Tensor<R>) -> ArithmeticResult<Tensor<R>> {
        self.check_capture(value)?;
        let captured = Tensor::empty(value.shape(), value.dtype(), self.client().device())?;
        self.client().copy_into(&captured, value)?;
        Ok(captured)
    }

    /// Check capture limits for inputs that a kernel snapshots into fresh storage.
    fn check_capture(&self, value: &Tensor<R>) -> ArithmeticResult<()> {
        self.arithmetic.tensor_metadata(value)?;
        u32::try_from(value.numel()).map_err(|_| Error::InvalidArgument {
            arg: "interval",
            reason: format!(
                "Interval capture count {} exceeds u32::MAX. Split the input batch.",
                value.numel()
            ),
        })?;
        value
            .numel()
            .checked_mul(value.dtype().size_in_bytes())
            .ok_or_else(|| Error::InvalidArgument {
                arg: "interval",
                reason: format!(
                    "Interval capture shape {:?} overflows allocation bytes. Split the input batch.",
                    value.shape()
                ),
            })?;
        Ok(())
    }
}

#[path = "context/elementary.rs"]
mod elementary;
#[path = "context/expansion_add.rs"]
mod expansion_add;
#[path = "context/expansion_division.rs"]
mod expansion_division;
#[path = "context/expansion_inputs.rs"]
mod expansion_inputs;
#[path = "context/expansion_interval.rs"]
mod expansion_interval;
#[path = "context/expansion_lift.rs"]
mod expansion_lift;
#[path = "context/expansion_product.rs"]
mod expansion_product;
#[path = "context/expansion_sqrt.rs"]
mod expansion_sqrt;
#[path = "context/positive_product_ratio.rs"]
mod positive_product_ratio;
#[path = "context/scaled_product_format.rs"]
mod scaled_product_format;
#[path = "context/scaled_product_normalize.rs"]
mod scaled_product_normalize;
#[path = "context/scaled_product_output.rs"]
mod scaled_product_output;
