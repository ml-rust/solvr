//! Initialize and resume one qualified, owned root session.

use super::super::certified::{
    metadata,
    native::{self, RootClient},
};
use super::super::certified_bisection::{admission, classification};
use super::{selection, state};
use crate::DType;
use crate::common::arithmetic::{IntervalContext, with_interval_arithmetic};
use crate::optimize::OptimizeResult;
use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedMappedRootProgress, CertifiedRootAdvanceAllowance,
    CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
    CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
};
use numr::ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn bisect_certified_mapped_impl<R, C, F, M>(
    client: &C,
    residual: F,
    mapping: M,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative_lower: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    admission::metadata(client, lower, upper, derivative_lower, options)?;
    with_interval_arithmetic(
        client,
        lower.dtype(),
        "certified mapped increasing root",
        |ctx| {
            Ok(initialize(
                ctx,
                &residual,
                &mapping,
                lower,
                upper,
                derivative_lower,
                options,
            ))
        },
    )
    .map_err(native::arithmetic)?
}

fn initialize<R, C, F, M>(
    ctx: &mut IntervalContext<'_, R, C>,
    residual: &F,
    mapping: &M,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    let client = ctx.client();
    let lower = native::owned(client, lower)?;
    let upper = native::owned(client, upper)?;
    let derivative = native::owned(client, derivative)?;
    let state = if lower.numel() == 0 {
        state::empty(client, lower, upper, derivative)?
    } else {
        admission::values(client, &lower, &upper, &derivative)?;
        classification::endpoints(client, residual, &lower, &upper, &derivative)?
    };
    let floor = Tensor::zeros(state.owner.shape(), state.lower.dtype(), client.device())?;
    let continuation = CertifiedRootContinuation {
        state,
        floor,
        max_iterations: options.max_iterations,
        max_points: options.max_points,
        iterations: 0,
        mapped_points: 0,
    };
    state::advance(
        ctx,
        residual,
        mapping,
        continuation,
        &CertifiedRootTolerance {
            atol: options.atol,
            rtol: options.rtol,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: options.max_iterations,
            max_points: options.max_points,
        },
        None,
    )
}

pub fn resume_certified_mapped_impl<R, C, F, M>(
    client: &C,
    residual: F,
    mapping: M,
    continuation: CertifiedRootContinuation<R>,
    tolerance: &CertifiedRootTolerance,
    allowance: &CertifiedRootAdvanceAllowance,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    resume(
        client,
        residual,
        mapping,
        continuation,
        ResumeSelection::All,
        tolerance,
        allowance,
    )
}

pub fn resume_certified_mapped_selected_impl<R, C, F, M>(
    client: &C,
    residual: F,
    mapping: M,
    continuation: CertifiedRootContinuation<R>,
    owners: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
    allowance: &CertifiedRootAdvanceAllowance,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    resume(
        client,
        residual,
        mapping,
        continuation,
        ResumeSelection::Selected(owners),
        tolerance,
        allowance,
    )
}

pub fn refresh_certified_mapped_selected_impl<R, C, F, M>(
    client: &C,
    residual: F,
    mapping: M,
    continuation: CertifiedRootContinuation<R>,
    owners: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
    allowance: &CertifiedRootAdvanceAllowance,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    resume(
        client,
        residual,
        mapping,
        continuation,
        ResumeSelection::RefreshSelected(owners),
        tolerance,
        allowance,
    )
}

enum ResumeSelection<'a, R: Runtime<DType = DType>> {
    All,
    Selected(&'a Tensor<R>),
    RefreshSelected(&'a Tensor<R>),
}

fn resume<R, C, F, M>(
    client: &C,
    residual: F,
    mapping: M,
    mut continuation: CertifiedRootContinuation<R>,
    selection: ResumeSelection<'_, R>,
    tolerance: &CertifiedRootTolerance,
    allowance: &CertifiedRootAdvanceAllowance,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    let lower = &continuation.state.lower;
    metadata::tensor(
        client,
        lower,
        lower.numel(),
        lower.dtype(),
        "continuation lower",
    )?;
    let (owners, refresh) = match selection {
        ResumeSelection::All => (None, false),
        ResumeSelection::Selected(owners) => (Some(owners), false),
        ResumeSelection::RefreshSelected(owners) => (Some(owners), true),
    };
    let demand = owners
        .map(|owners| selection::demand(client, &continuation, owners))
        .transpose()?;
    admission::tolerances(tolerance.atol, tolerance.rtol)?;
    let dtype = lower.dtype();
    with_interval_arithmetic(client, dtype, "certified mapped root continuation", |ctx| {
        if let (true, Some(demand)) = (refresh, demand.as_ref())
            && let Err(error) = selection::refresh(client, &mut continuation, demand)
        {
            return Ok(Err(error));
        }
        Ok(state::advance(
            ctx,
            &residual,
            &mapping,
            continuation,
            tolerance,
            allowance,
            demand.as_ref(),
        ))
    })
    .map_err(native::arithmetic)?
}
