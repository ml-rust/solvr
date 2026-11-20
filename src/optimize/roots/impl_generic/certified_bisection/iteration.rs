//! Compact probe rounds and freeze completed root enclosures.

use super::classification::State;
use super::native::{self, RootClient};
use super::{
    evaluate, extended,
    probe::{self, Probe},
    stopping::{Identity, Stopping},
};
use crate::DType;
use crate::common::arithmetic::{Interval, IntervalContext};
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
        let remaining = options.max_points.saturating_sub(state.evaluated);
        if pending.numel() > remaining {
            break;
        }
        let lower = client.index_select(&state.lower, 0, &pending)?;
        let upper = client.index_select(&state.upper, 0, &pending)?;
        let rows = state
            .probe_state
            .as_ref()
            .map(|rows| client.index_select(rows, 0, &pending))
            .transpose()?;
        let probe::Selection { point, probe } = probe::select(client, rows, &lower, &upper)?;
        // Second points fit in row order into the points left after this round's midpoints.
        // Nothing is held back for later rounds, so a limit that never binds keeps the
        // unbounded run's path bit for bit. A binding limit can run fewer rounds than
        // midpoint probes. Its rows still report `WorkLimit` with certified brackets.
        let probe = probe
            .map(|probe| limit(client, probe, remaining - pending.numel()))
            .transpose()?;
        rounds += 1;
        let outside = client.maximum(&client.lt(&point, &lower)?, &client.gt(&point, &upper)?)?;
        if native::any(
            client,
            &client.maximum(&outside, &native::nonfinite(client, &point)?)?,
        )? {
            return Err(OptimizeError::NumericalError {
                message: "Certified root probe is nonfinite or outside its bracket. Check the runtime midpoint and rounding implementations.".into(),
            });
        }
        let interior = client.mul(&client.gt(&point, &lower)?, &client.lt(&point, &upper)?)?;
        let previous = client.index_select(&state.previous_point, 0, &pending)?;
        let sampled = client.index_select(&state.sampled, 0, &pending)?;
        let repeated = client.mul(&sampled, &client.eq(&point, &previous)?)?;
        // Only a repeated unpaired midpoint probe shows that no further contraction exists.
        // A paired row still evaluates a new second point.
        let repeated = match probe.as_ref() {
            Some(probe) => client.mul(
                &client.mul(&repeated, &probe.midpoint)?,
                &client.sub(
                    &Tensor::ones(probe.paired.shape(), probe.paired.dtype(), client.device())?,
                    &probe.paired,
                )?,
            )?,
            None => repeated,
        };
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
        let probe = probe
            .map(|probe| probe.compact(client, &interior))
            .transpose()?;
        contract(
            ctx,
            f,
            state,
            Round {
                roots: &roots,
                point: &point,
                probe: probe.as_ref(),
            },
            options,
            policy,
        )?;
    }
    Ok(rounds)
}

/// Keep at most `count` paired rows, the first in row order.
fn limit<R, C>(client: &C, probe: Probe<R>, count: usize) -> OptimizeResult<Probe<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    // A row pairs at most once, so a count at the row count keeps every pair.
    if count >= probe.paired.numel() {
        return Ok(probe);
    }
    let rank = client.cumsum(&probe.paired, 0)?;
    let kept = client.le(
        &rank,
        &Tensor::full_scalar(rank.shape(), rank.dtype(), count as f64, client.device())?,
    )?;
    Ok(Probe {
        paired: client.mul(&probe.paired, &kept)?,
        ..probe
    })
}

/// Evaluated roots of one round, their probes, and their probe rows.
struct Round<'a, R: Runtime<DType = DType>> {
    roots: &'a Tensor<R>,
    point: &'a Tensor<R>,
    probe: Option<&'a Probe<R>>,
}

fn contract<R, C, F, P>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    state: &mut State<R>,
    round: Round<'_, R>,
    options: &CertifiedBisectionOptions,
    policy: &mut P,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
    P: Stopping<R, C>,
{
    let Round {
        roots,
        point,
        probe,
    } = round;
    let client = ctx.client();
    let lower = client.index_select(&state.lower, 0, roots)?;
    let upper = client.index_select(&state.upper, 0, roots)?;
    let owners = client.index_select(&state.owner, 0, roots)?;
    let derivative = client.index_select(&state.derivative, 0, roots)?;
    let pairs = match probe {
        Some(probe) => Some(native::indices(client, &probe.paired)?).filter(|p| p.numel() > 0),
        None => None,
    };
    let (residual, paired) = match (probe, pairs.as_ref()) {
        (Some(probe), Some(pairs)) => {
            // One callback serves the midpoint of every row and the secant point of each pair.
            let count = roots.numel();
            let second = client.index_select(&probe.second, 0, pairs)?;
            let all = evaluate::callback(
                client,
                f,
                &client.cat(&[&owners, &client.index_select(&owners, 0, pairs)?], 0)?,
                &client.cat(&[point, &second], 0)?,
            )?;
            let part = |value: &Tensor<R>, start: usize, length: usize| {
                native::owned(client, &value.narrow(0, start, length)?)
            };
            let residual = CertifiedRootValues {
                lower: part(&all.lower, 0, count)?,
                upper: part(&all.upper, 0, count)?,
            };
            let paired = CertifiedRootValues {
                lower: part(&all.lower, count, pairs.numel())?,
                upper: part(&all.upper, count, pairs.numel())?,
            };
            (residual, Some((pairs, second, paired)))
        }
        _ => (evaluate::callback(client, f, &owners, point)?, None),
    };
    state.evaluated += roots.numel() + pairs.as_ref().map_or(0, Tensor::numel);
    let zero = Tensor::zeros(point.shape(), point.dtype(), client.device())?;
    let corrected = correct(ctx, [&lower, &upper], point, &residual, &derivative)?;
    let (mut new_lower, mut new_upper) = (corrected.lower().clone(), corrected.upper().clone());
    let mut second = None;
    if let (Some(probe), Some((pairs, points, residual))) = (probe, paired.as_ref()) {
        let gather = |value: &Tensor<R>| client.index_select(value, 0, pairs);
        let other = correct(
            ctx,
            [&gather(&lower)?, &gather(&upper)?],
            points,
            residual,
            &gather(&derivative)?,
        )?;
        // Both corrected brackets enclose the same root, so their intersection does too.
        let joined_lower = client.maximum(&gather(&new_lower)?, other.lower())?;
        let joined_upper = client.minimum(&gather(&new_upper)?, other.upper())?;
        if native::any(client, &client.gt(&joined_lower, &joined_upper)?)? {
            return Err(OptimizeError::NumericalError {
                message: "Certified root residuals at a midpoint and a secant point contradict increasing order. Correct the callback or derivative assertion.".into(),
            });
        }
        new_lower = native::replace(client, &new_lower, pairs, &joined_lower)?;
        new_upper = native::replace(client, &new_upper, pairs, &joined_upper)?;
        let spread = |value: &Tensor<R>| native::replace(client, &zero, pairs, value);
        second = Some((probe, spread(&residual.lower)?, spread(&residual.upper)?));
    }
    let accepted = policy.satisfied(ctx, &owners, &new_lower, &new_upper, options)?;
    let unchanged = client.mul(
        &client.eq(&lower, &new_lower)?,
        &client.eq(&upper, &new_upper)?,
    )?;
    // A stalled interpolated probe keeps its row active. The gate keeps the schedule intact.
    // A first midpoint stall with an untried simplest float also keeps its row active.
    // The next round evaluates that float, and a second stall sets the floor.
    let unchanged = match probe {
        Some(probe) => client.mul(
            &client.mul(&unchanged, &probe.midpoint)?,
            &client.sub(
                &Tensor::ones(point.shape(), point.dtype(), client.device())?,
                &probe.retry,
            )?,
        )?,
        None => unchanged,
    };
    let unchanged = if policy.parameter_singleton_floor() {
        client.maximum(&unchanged, &client.eq(&new_lower, &new_upper)?)?
    } else {
        unchanged
    };
    let stalled = client.mul(&unchanged, &client.eq(&accepted, &zero)?)?;
    let converged = client.index_select(roots, 0, &native::indices(client, &accepted)?)?;
    let floors = client.index_select(roots, 0, &native::indices(client, &stalled)?)?;
    status(client, state, &converged, 0.0)?;
    status(client, state, &floors, 3.0)?;
    state.lower = native::replace(client, &state.lower, roots, &new_lower)?;
    state.upper = native::replace(client, &state.upper, roots, &new_upper)?;
    let active = Tensor::ones(point.shape(), point.dtype(), client.device())?;
    let remaining = client.sub(&active, &client.maximum(&accepted, &stalled)?)?;
    state.active = native::replace(client, &state.active, roots, &remaining)?;
    state.previous_point = native::replace(client, &state.previous_point, roots, point)?;
    state.sampled = native::replace(client, &state.sampled, roots, &active)?;
    if let (Some(probe), Some(all)) = (probe, state.probe_state.as_ref()) {
        let rows = probe::record(
            client,
            &probe.rows,
            point,
            [&residual.lower, &residual.upper],
            [&new_lower, &new_upper],
            second.as_ref().map(|(probe, lower, upper)| probe::Second {
                paired: &probe.paired,
                point: &probe.second,
                residual: [lower, upper],
            }),
        )?;
        state.probe_state = Some(client.index_put(all, 0, roots, &rows)?);
    }
    Ok(())
}

/// Cut a bracket at one evaluated point per row, then apply the slope correction.
fn correct<R, C>(
    ctx: &mut IntervalContext<'_, R, C>,
    bracket: [&Tensor<R>; 2],
    point: &Tensor<R>,
    residual: &CertifiedRootValues<R>,
    derivative: &Tensor<R>,
) -> OptimizeResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let client = ctx.client();
    let [lower, upper] = bracket;
    let zero = Tensor::zeros(point.shape(), point.dtype(), client.device())?;
    let negative = client.lt(&residual.upper, &zero)?;
    let positive = client.gt(&residual.lower, &zero)?;
    let exact = client.mul(
        &client.eq(&residual.lower, &zero)?,
        &client.eq(&residual.upper, &zero)?,
    )?;
    let cut_lower = client.where_cond(&client.maximum(&negative, &exact)?, point, lower)?;
    let cut_upper = client.where_cond(&client.maximum(&positive, &exact)?, point, upper)?;
    let active = Tensor::ones(point.shape(), point.dtype(), client.device())?;
    extended::intersect(
        ctx,
        extended::CorrectionInputs {
            lower: &cut_lower,
            upper: &cut_upper,
            point,
            residual_lower: &residual.lower,
            residual_upper: &residual.upper,
            derivative_lower: derivative,
            active: &active,
        },
    )
    .map_err(native::arithmetic)
}
