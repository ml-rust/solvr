use super::super::certified::native::RootClient;
use super::state::State;
use crate::DType;
use crate::optimize::OptimizeResult;
use crate::optimize::roots::traits::CertifiedUpperBracketStatus;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn classify<R, C>(
    client: &C,
    state: &mut State<R>,
    limit: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let zero = Tensor::zeros(state.lower.shape(), state.lower.dtype(), client.device())?;
    let lower_exact = client.mul(
        &client.eq(&state.lower_residual.lower, &zero)?,
        &client.eq(&state.lower_residual.upper, &zero)?,
    )?;
    let upper_exact = client.mul(
        &client.eq(&state.upper_residual.lower, &zero)?,
        &client.eq(&state.upper_residual.upper, &zero)?,
    )?;
    let bracket = client.mul(
        &client.le(&state.lower_residual.upper, &zero)?,
        &client.ge(&state.upper_residual.lower, &zero)?,
    )?;
    let bracket = client.maximum(&bracket, &client.maximum(&lower_exact, &upper_exact)?)?;
    state.finish(client, &bracket, CertifiedUpperBracketStatus::Bracketed)?;
    let at_limit = client.eq(&state.upper, limit)?;
    let absent = client.maximum(
        &client.gt(&state.lower_residual.lower, &zero)?,
        &client.mul(&at_limit, &client.lt(&state.upper_residual.upper, &zero)?)?,
    )?;
    state.finish(client, &absent, CertifiedUpperBracketStatus::NoRoot)?;
    state.finish(client, &at_limit, CertifiedUpperBracketStatus::Unresolved)?;
    Ok(())
}
