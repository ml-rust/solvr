//! Preserve numerical state and isolate externally visible snapshots.

use super::super::certified::native::{self, RootClient};
use super::super::certified_bisection::{classification::State, iteration};
use super::policy::Mapped;
use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedMappedRootProgress, CertifiedRootAdvanceAllowance,
    CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
    CertifiedRootQuery, CertifiedRootResult, CertifiedRootStatus, CertifiedRootTolerance,
    CertifiedRootValues,
};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn empty<R, C>(
    client: &C,
    lower: Tensor<R>,
    upper: Tensor<R>,
    derivative: Tensor<R>,
) -> OptimizeResult<State<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let dtype = lower.dtype();
    Ok(State {
        status: Tensor::empty(&[0], DType::I64, client.device())?,
        owner: Tensor::empty(&[0], DType::I64, client.device())?,
        lower,
        upper,
        derivative,
        active: Tensor::empty(&[0], dtype, client.device())?,
        evaluated: 0,
        previous_point: Tensor::empty(&[0], dtype, client.device())?,
        sampled: Tensor::empty(&[0], dtype, client.device())?,
    })
}

pub(super) fn advance<R, C, F, M>(
    ctx: &mut IntervalContext<'_, R, C>,
    residual: &F,
    mapping: &M,
    mut continuation: CertifiedRootContinuation<R>,
    tolerance: &CertifiedRootTolerance,
    allowance: &CertifiedRootAdvanceAllowance,
    demand: Option<&Tensor<R>>,
) -> OptimizeResult<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    let remaining_rounds = continuation
        .max_iterations
        .checked_sub(continuation.iterations)
        .ok_or_else(counter_error)?;
    let remaining_points = continuation
        .max_points
        .checked_sub(continuation.state.evaluated)
        .ok_or_else(counter_error)?;
    let point_limit = continuation
        .state
        .evaluated
        .checked_add(remaining_points.min(allowance.max_points))
        .ok_or_else(counter_error)?;
    let options = CertifiedBisectionOptions {
        atol: tolerance.atol,
        rtol: tolerance.rtol,
        max_iterations: remaining_rounds.min(allowance.max_iterations),
        max_points: point_limit,
    };
    let mut policy = Mapped {
        callback: mapping,
        points: continuation.mapped_points,
    };
    let rounds = iteration::run_policy(
        ctx,
        residual,
        &mut continuation.state,
        &options,
        &mut policy,
        Some(&continuation.floor),
        demand,
    )?;
    continuation.iterations = continuation
        .iterations
        .checked_add(rounds)
        .ok_or_else(counter_error)?;
    continuation.mapped_points = policy.points;
    let client = ctx.client();
    let code = Tensor::full_scalar(
        &[],
        DType::I64,
        CertifiedRootStatus::AccuracyFloor as i64 as f64,
        client.device(),
    )?;
    let state = &continuation.state;
    if let Some(mask) = demand {
        let roots = native::indices(client, mask)?;
        if roots.numel() != 0 {
            let owner = client.index_select(&state.owner, 0, &roots)?;
            let status = client.index_select(&state.status, 0, &owner)?;
            let floor = client.cast(&client.eq(&status, &code)?, state.lower.dtype())?;
            let previous = client.index_select(&continuation.floor, 0, &roots)?;
            let floor = client.maximum(&previous, &floor)?;
            continuation.floor = native::replace(client, &continuation.floor, &roots, &floor)?;
        }
    } else {
        let status = client.index_select(&state.status, 0, &state.owner)?;
        let floor = client.cast(&client.eq(&status, &code)?, state.lower.dtype())?;
        continuation.floor = client.maximum(&continuation.floor, &floor)?;
    }
    let result = snapshot(client, &continuation)?;
    Ok(CertifiedMappedRootProgress {
        result,
        continuation,
    })
}

fn snapshot<R, C>(
    client: &C,
    continuation: &CertifiedRootContinuation<R>,
) -> OptimizeResult<CertifiedRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let state = &continuation.state;
    Ok(CertifiedRootResult {
        status: native::owned(client, &state.status)?,
        root_owner: native::owned(client, &state.owner)?,
        root_lower: native::owned(client, &state.lower)?,
        root_upper: native::owned(client, &state.upper)?,
        iterations: continuation.iterations,
        evaluated_points: state.evaluated,
    })
}

fn counter_error() -> OptimizeError {
    OptimizeError::NumericalError {
        message: "Certified root continuation counters exceed their original ceilings. Discard the continuation.".into(),
    }
}
