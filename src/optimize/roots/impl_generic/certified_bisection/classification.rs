//! Establish physical root existence before creating compact root state.

use super::native::RootClient;
use super::{evaluate, native, probe};
use crate::DType;
use crate::optimize::roots::traits::{CertifiedRootProbe, CertifiedRootQuery, CertifiedRootValues};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) struct State<R: Runtime<DType = DType>> {
    pub status: Tensor<R>,
    pub owner: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub derivative: Tensor<R>,
    pub active: Tensor<R>,
    pub evaluated: usize,
    pub previous_point: Tensor<R>,
    pub sampled: Tensor<R>,
    pub probe_state: Option<Tensor<R>>,
}

pub(crate) fn endpoints<R, C, F>(
    client: &C,
    f: &F,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    derivative: &Tensor<R>,
    probe: CertifiedRootProbe,
) -> OptimizeResult<State<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    let len = lower.numel();
    let dtype = lower.dtype();
    let zero = Tensor::zeros(&[len], dtype, client.device())?;
    let one = Tensor::ones(&[len], dtype, client.device())?;
    let owners = client.arange(0.0, len as f64, 1.0, DType::I64)?;
    let left = evaluate::callback(client, f, &owners, lower)?;
    let left_exact = client.mul(
        &client.eq(&left.lower, &zero)?,
        &client.eq(&left.upper, &zero)?,
    )?;
    let left_absent = client.gt(&left.lower, &zero)?;
    let distinct = client.lt(lower, upper)?;
    let pending = client.mul(
        &distinct,
        &client.sub(&one, &client.maximum(&left_exact, &left_absent)?)?,
    )?;
    let queried = native::indices(client, &pending)?;
    let mut right_lower = native::owned(client, &left.lower)?;
    let mut right_upper = native::owned(client, &left.upper)?;
    if queried.numel() > 0 {
        let points = client.index_select(upper, 0, &queried)?;
        let query_owners = client.index_select(&owners, 0, &queried)?;
        let right = evaluate::callback(client, f, &query_owners, &points)?;
        let previous_lower = client.index_select(&left.lower, 0, &queried)?;
        if native::any(client, &client.gt(&previous_lower, &right.upper)?)? {
            return Err(OptimizeError::NumericalError {
                message: "Certified root endpoint residuals contradict increasing order. Correct the callback or derivative assertion.".into(),
            });
        }
        right_lower = native::replace(client, &right_lower, &queried, &right.lower)?;
        right_upper = native::replace(client, &right_upper, &queried, &right.upper)?;
    }
    let right_exact = client.mul(
        &pending,
        &client.mul(
            &client.eq(&right_lower, &zero)?,
            &client.eq(&right_upper, &zero)?,
        )?,
    )?;
    let bracket = client.mul(
        &pending,
        &client.mul(
            &client.le(&left.upper, &zero)?,
            &client.ge(&right_lower, &zero)?,
        )?,
    )?;
    let exists = client.maximum(&left_exact, &client.maximum(&right_exact, &bracket)?)?;
    let absent = client.maximum(&left_absent, &client.lt(&right_upper, &zero)?)?;
    let unresolved_code = Tensor::full_scalar(&[len], DType::I64, 2.0, client.device())?;
    let absent_code = Tensor::full_scalar(&[len], DType::I64, 1.0, client.device())?;
    let root_code = Tensor::full_scalar(&[len], DType::I64, 4.0, client.device())?;
    let status = client.where_cond(
        &client.cast(&absent, DType::U8)?,
        &absent_code,
        &unresolved_code,
    )?;
    let status = client.where_cond(&client.cast(&exists, DType::U8)?, &root_code, &status)?;
    let root_indices = native::indices(client, &exists)?;
    let collapsed_lower = client.where_cond(&right_exact, upper, lower)?;
    let collapsed_upper = client.where_cond(&left_exact, lower, upper)?;
    let owner = client.index_select(&owners, 0, &root_indices)?;
    let lower = client.index_select(&collapsed_lower, 0, &root_indices)?;
    let upper = client.index_select(&collapsed_upper, 0, &root_indices)?;
    let derivative = client.index_select(derivative, 0, &root_indices)?;
    let active = Tensor::ones(&[root_indices.numel()], dtype, client.device())?;
    let previous_point = Tensor::zeros(&[root_indices.numel()], dtype, client.device())?;
    let sampled = Tensor::zeros(&[root_indices.numel()], dtype, client.device())?;
    let probe_state = match probe {
        CertifiedRootProbe::Midpoint => None,
        CertifiedRootProbe::SlackSecant => {
            let gather = |value: &Tensor<R>| client.index_select(value, 0, &root_indices);
            Some(probe::seed(
                client,
                &lower,
                &upper,
                [&gather(&left.lower)?, &gather(&left.upper)?],
                [&gather(&right_lower)?, &gather(&right_upper)?],
            )?)
        }
    };
    Ok(State {
        status,
        owner,
        lower,
        upper,
        derivative,
        active,
        evaluated: len + queried.numel(),
        previous_point,
        sampled,
        probe_state,
    })
}
