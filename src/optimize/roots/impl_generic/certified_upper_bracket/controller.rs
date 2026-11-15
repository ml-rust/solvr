use super::super::certified::{evaluate, native};
use super::state::State;
use super::{admission, candidate, classification};
use crate::DType;
use crate::common::arithmetic::{IntervalContext, with_interval_arithmetic};
use crate::optimize::roots::traits::{
    CertifiedRootQuery, CertifiedRootValues, CertifiedUpperBracketOptions,
    CertifiedUpperBracketResult, CertifiedUpperBracketStatus,
};
use crate::optimize::{OptimizeError, OptimizeResult};
use native::RootClient;
use numr::ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn bracket_upper_certified_impl<R, C, F>(
    client: &C,
    f: F,
    lower: &Tensor<R>,
    initial_upper: &Tensor<R>,
    upper_limit: &Tensor<R>,
    options: &CertifiedUpperBracketOptions,
) -> OptimizeResult<CertifiedUpperBracketResult<R>>
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
{
    let initial = admission::metadata(client, lower, initial_upper, upper_limit, options)?;
    with_interval_arithmetic(client, lower.dtype(), "certified upper bracket", |ctx| {
        Ok(solve(
            ctx,
            &f,
            lower,
            initial_upper,
            upper_limit,
            options,
            initial,
        ))
    })
    .map_err(native::arithmetic)?
}

fn solve<R, C, F>(
    ctx: &mut IntervalContext<'_, R, C>,
    f: &F,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    limit: &Tensor<R>,
    options: &CertifiedUpperBracketOptions,
    initial: usize,
) -> OptimizeResult<CertifiedUpperBracketResult<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    let client = ctx.client();
    let lower = native::owned(client, lower)?;
    let upper = native::owned(client, upper)?;
    let limit = native::owned(client, limit)?;
    if lower.numel() == 0 {
        return Ok(CertifiedUpperBracketResult {
            lower_residual: CertifiedRootValues {
                lower: Tensor::empty(&[0], lower.dtype(), client.device())?,
                upper: Tensor::empty(&[0], lower.dtype(), client.device())?,
            },
            upper_residual: CertifiedRootValues {
                lower: Tensor::empty(&[0], lower.dtype(), client.device())?,
                upper: Tensor::empty(&[0], lower.dtype(), client.device())?,
            },
            lower,
            upper,
            status: Tensor::empty(&[0], DType::I64, client.device())?,
            expansions: Tensor::empty(&[0], DType::I64, client.device())?,
            expansion_rounds: 0,
            callback_points: 0,
        });
    }
    admission::values(client, &lower, &upper, &limit)?;
    let owners = client.arange(0.0, lower.numel() as f64, 1.0, DType::I64)?;
    // Both initial endpoints are physical queries, including coincident endpoints.
    let left = evaluate::callback(client, f, &owners, &lower)?;
    let right = evaluate::callback(client, f, &owners, &upper)?;
    let mut state = State::new(client, lower, upper, left, right, owners, initial)?;
    classification::classify(client, &mut state, &limit)?;
    while state.expansion_rounds < options.max_expansions {
        let pending = native::indices(client, &state.active)?;
        if pending.numel() == 0 {
            break;
        }
        let lower = client.index_select(&state.lower, 0, &pending)?;
        let upper = client.index_select(&state.upper, 0, &pending)?;
        let cap = client.index_select(&limit, 0, &pending)?;
        let points = candidate::expanded(client, &lower, &upper, &cap)?;
        let progress = client.gt(&points, &upper)?;
        let zero = Tensor::zeros(progress.shape(), progress.dtype(), client.device())?;
        let stalled = client.index_select(
            &pending,
            0,
            &native::indices(client, &client.eq(&progress, &zero)?)?,
        )?;
        if stalled.numel() > 0 {
            let mask = Tensor::zeros(state.active.shape(), state.active.dtype(), client.device())?;
            let ones = Tensor::ones(&[stalled.numel()], state.active.dtype(), client.device())?;
            let mask = native::replace(client, &mask, &stalled, &ones)?;
            state.finish(client, &mask, CertifiedUpperBracketStatus::Unresolved)?;
        }
        let progressing = native::indices(client, &progress)?;
        if progressing.numel() == 0 {
            break;
        }
        let next_count = state.callback_points.checked_add(progressing.numel()).ok_or_else(|| {
            OptimizeError::InvalidInput {
                context: "Certified upper bracket callback count overflows usize. Reduce the work budget.".into(),
            }
        })?;
        if next_count > options.max_points {
            break;
        }
        let rows = client.index_select(&pending, 0, &progressing)?;
        let points = client.index_select(&points, 0, &progressing)?;
        let owners = client.index_select(&state.owners, 0, &rows)?;
        let residual = evaluate::callback(client, f, &owners, &points)?;
        state.upper = native::replace(client, &state.upper, &rows, &points)?;
        state.upper_residual.lower =
            native::replace(client, &state.upper_residual.lower, &rows, &residual.lower)?;
        state.upper_residual.upper =
            native::replace(client, &state.upper_residual.upper, &rows, &residual.upper)?;
        let counts = client.index_select(&state.expansions, 0, &rows)?;
        let one = Tensor::ones(counts.shape(), DType::I64, client.device())?;
        let counts = client.add(&counts, &one)?;
        state.expansions = native::replace(client, &state.expansions, &rows, &counts)?;
        state.callback_points = next_count;
        state.expansion_rounds += 1;
        classification::classify(client, &mut state, &limit)?;
    }
    Ok(CertifiedUpperBracketResult {
        lower: state.lower,
        upper: state.upper,
        lower_residual: state.lower_residual,
        upper_residual: state.upper_residual,
        status: state.status,
        expansions: state.expansions,
        expansion_rounds: state.expansion_rounds,
        callback_points: state.callback_points,
    })
}
