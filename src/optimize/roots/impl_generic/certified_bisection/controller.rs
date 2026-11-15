//! Coordinate admitted native increasing-root calculations.

use super::native::{self, RootClient};
use super::{admission, classification, iteration};
use crate::DType;
use crate::common::arithmetic::{IntervalContext, with_interval_arithmetic};
use crate::optimize::OptimizeResult;
use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedRootQuery, CertifiedRootResult, CertifiedRootValues,
};
use numr::ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn bisect_certified_impl<R, C, F>(
    client: &C,
    f: F,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative_lower: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<CertifiedRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    admission::metadata(client, lower, upper, derivative_lower, options)?;
    // The nested result keeps arithmetic completion mandatory for every successful solver return.
    with_interval_arithmetic(client, lower.dtype(), "certified increasing root", |ctx| {
        Ok(solve(ctx, &f, lower, upper, derivative_lower, options))
    })
    .map_err(native::arithmetic)?
}

fn solve<R, C, F>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<CertifiedRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    let client = ctx.client();
    let lower = native::owned(client, lower)?;
    let upper = native::owned(client, upper)?;
    let derivative = native::owned(client, derivative)?;
    if lower.numel() == 0 {
        return Ok(CertifiedRootResult {
            status: Tensor::empty(&[0], DType::I64, client.device())?,
            root_owner: Tensor::empty(&[0], DType::I64, client.device())?,
            root_lower: lower,
            root_upper: upper,
            iterations: 0,
            evaluated_points: 0,
        });
    }
    admission::values(client, &lower, &upper, &derivative)?;
    let mut state = classification::endpoints(client, f, &lower, &upper, &derivative)?;
    let iterations = iteration::run(ctx, f, &mut state, options)?;
    Ok(CertifiedRootResult {
        status: native::owned(client, &state.status)?,
        root_owner: native::owned(client, &state.owner)?,
        root_lower: native::owned(client, &state.lower)?,
        root_upper: native::owned(client, &state.upper)?,
        iterations,
        evaluated_points: state.evaluated,
    })
}
