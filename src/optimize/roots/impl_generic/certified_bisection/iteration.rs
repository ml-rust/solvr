//! Compact midpoint rounds and freeze completed root enclosures.

use super::classification::State;
use super::native::{self, RootClient};
use super::{
    evaluate, extended,
    stopping::{Identity, Stopping},
};
use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedRootQuery, CertifiedRootValues,
};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

fn status<R, C>(
    client: &C,
    state: &mut State<R>,
    roots: &Tensor<R>,
    code: f64,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    if roots.numel() == 0 {
        return Ok(());
    }
    let owners = client.index_select(&state.owner, 0, roots)?;
    let codes = Tensor::full_scalar(&[roots.numel()], DType::I64, code, client.device())?;
    state.status = native::replace(client, &state.status, &owners, &codes)?;
    Ok(())
}

fn root_indices<R, C>(
    client: &C,
    roots: Option<&Tensor<R>>,
    mask: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let indices = native::indices(client, mask)?;
    match roots {
        Some(roots) => Ok(client.index_select(roots, 0, &indices)?),
        None => Ok(indices),
    }
}

pub(super) fn run<R, C, F>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    state: &mut State<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<usize>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    run_policy(ctx, f, state, options, &mut Identity, None, None)
}

pub(crate) fn run_policy<R, C, F, P>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    state: &mut State<R>,
    options: &CertifiedBisectionOptions,
    policy: &mut P,
    permanent_floor: Option<&Tensor<R>>,
    demand: Option<&Tensor<R>>,
) -> OptimizeResult<usize>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    P: Stopping<R, C>,
{
    let client = ctx.client();
    if state.owner.numel() == 0 {
        return Ok(0);
    }
    {
        let roots = demand
            .map(|mask| native::indices(client, mask))
            .transpose()?;
        if roots.as_ref().is_some_and(|roots| roots.numel() == 0) {
            return Ok(0);
        }
        let (owner, lower, upper) = match roots.as_ref() {
            Some(roots) => (
                client.index_select(&state.owner, 0, roots)?,
                client.index_select(&state.lower, 0, roots)?,
                client.index_select(&state.upper, 0, roots)?,
            ),
            None => (
                state.owner.clone(),
                state.lower.clone(),
                state.upper.clone(),
            ),
        };
        let accepted = policy.satisfied(ctx, &owner, &lower, &upper, options)?;
        let converged = root_indices(client, roots.as_ref(), &accepted)?;
        status(client, state, &converged, 0.0)?;
        let one = Tensor::ones(accepted.shape(), accepted.dtype(), client.device())?;
        let mut active = client.sub(&one, &accepted)?;
        if let Some(floor) = permanent_floor {
            let floor = match roots.as_ref() {
                Some(roots) => client.index_select(floor, 0, roots)?,
                None => floor.clone(),
            };
            let fixed = client.maximum(&floor, &client.eq(&lower, &upper)?)?;
            let blocked = client.mul(&active, &fixed)?;
            let blocked = root_indices(client, roots.as_ref(), &blocked)?;
            status(client, state, &blocked, 3.0)?;
            active = client.mul(&active, &client.sub(&one, &fixed)?)?;
            let pending = root_indices(client, roots.as_ref(), &active)?;
            status(client, state, &pending, 4.0)?;
        }
        state.active = match roots.as_ref() {
            Some(roots) => native::replace(client, &state.active, roots, &active)?,
            None => active,
        };
    }
    let mut rounds = 0;
    while rounds < options.max_iterations {
        let pending = match demand {
            Some(mask) => native::indices(client, &client.mul(&state.active, mask)?)?,
            None => native::indices(client, &state.active)?,
        };
        if pending.numel() == 0 {
            break;
        }
        // Admission covers the full pending batch before point or gather allocation.
        if pending.numel() > options.max_points.saturating_sub(state.evaluated) {
            break;
        }
        let lower = client.index_select(&state.lower, 0, &pending)?;
        let upper = client.index_select(&state.upper, 0, &pending)?;
        let point = client.midpoint(&lower, &upper)?;
        rounds += 1;
        let outside = client.maximum(&client.lt(&point, &lower)?, &client.gt(&point, &upper)?)?;
        if native::any(
            client,
            &client.maximum(&outside, &native::nonfinite(client, &point)?)?,
        )? {
            return Err(OptimizeError::NumericalError {
                message: "Certified root midpoint is nonfinite or outside its bracket. Check the runtime midpoint implementation.".into(),
            });
        }
        let interior = client.mul(&client.gt(&point, &lower)?, &client.lt(&point, &upper)?)?;
        let previous = client.index_select(&state.previous_point, 0, &pending)?;
        let sampled = client.index_select(&state.sampled, 0, &pending)?;
        let repeated = client.mul(&sampled, &client.eq(&point, &previous)?)?;
        let zero = Tensor::zeros(interior.shape(), interior.dtype(), client.device())?;
        let interior = client.mul(&interior, &client.eq(&repeated, &zero)?)?;
        let floors = native::indices(client, &client.eq(&interior, &zero)?)?;
        let floors = client.index_select(&pending, 0, &floors)?;
        status(client, state, &floors, 3.0)?;
        let floor_zero = Tensor::zeros(&[floors.numel()], state.active.dtype(), client.device())?;
        state.active = native::replace(client, &state.active, &floors, &floor_zero)?;
        let interior = native::indices(client, &interior)?;
        if interior.numel() == 0 {
            break;
        }
        let roots = client.index_select(&pending, 0, &interior)?;
        let point = client.index_select(&point, 0, &interior)?;
        contract(ctx, f, state, &roots, &point, options, policy)?;
    }
    Ok(rounds)
}

fn contract<R, C, F, P>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    state: &mut State<R>,
    roots: &Tensor<R>,
    point: &Tensor<R>,
    options: &CertifiedBisectionOptions,
    policy: &mut P,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    P: Stopping<R, C>,
{
    let client = ctx.client();
    let lower = client.index_select(&state.lower, 0, roots)?;
    let upper = client.index_select(&state.upper, 0, roots)?;
    let owners = client.index_select(&state.owner, 0, roots)?;
    let derivative = client.index_select(&state.derivative, 0, roots)?;
    let residual = evaluate::callback(client, f, &owners, point)?;
    state.evaluated += roots.numel();
    let zero = Tensor::zeros(point.shape(), point.dtype(), client.device())?;
    let negative = client.lt(&residual.upper, &zero)?;
    let positive = client.gt(&residual.lower, &zero)?;
    let exact = client.mul(
        &client.eq(&residual.lower, &zero)?,
        &client.eq(&residual.upper, &zero)?,
    )?;
    let cut_lower = client.where_cond(&client.maximum(&negative, &exact)?, point, &lower)?;
    let cut_upper = client.where_cond(&client.maximum(&positive, &exact)?, point, &upper)?;
    let active = Tensor::ones(point.shape(), point.dtype(), client.device())?;
    let corrected = extended::intersect(
        ctx,
        extended::CorrectionInputs {
            lower: &cut_lower,
            upper: &cut_upper,
            point,
            residual_lower: &residual.lower,
            residual_upper: &residual.upper,
            derivative_lower: &derivative,
            active: &active,
        },
    )
    .map_err(native::arithmetic)?;
    let accepted = policy.satisfied(ctx, &owners, corrected.lower(), corrected.upper(), options)?;
    let unchanged = client.mul(
        &client.eq(&lower, corrected.lower())?,
        &client.eq(&upper, corrected.upper())?,
    )?;
    let unchanged = if policy.parameter_singleton_floor() {
        client.maximum(
            &unchanged,
            &client.eq(corrected.lower(), corrected.upper())?,
        )?
    } else {
        unchanged
    };
    let stalled = client.mul(&unchanged, &client.eq(&accepted, &zero)?)?;
    let converged = client.index_select(roots, 0, &native::indices(client, &accepted)?)?;
    let floors = client.index_select(roots, 0, &native::indices(client, &stalled)?)?;
    status(client, state, &converged, 0.0)?;
    status(client, state, &floors, 3.0)?;
    state.lower = native::replace(client, &state.lower, roots, corrected.lower())?;
    state.upper = native::replace(client, &state.upper, roots, corrected.upper())?;
    let remaining = client.sub(&active, &client.maximum(&accepted, &stalled)?)?;
    state.active = native::replace(client, &state.active, roots, &remaining)?;
    state.previous_point = native::replace(client, &state.previous_point, roots, point)?;
    state.sampled = native::replace(client, &state.sampled, roots, &active)?;
    Ok(())
}
