//! Check original-row demand and gather its compact-root mask.

use super::super::certified::{
    metadata,
    native::{self, RootClient},
};
use crate::DType;
use crate::optimize::roots::traits::CertifiedRootContinuation;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn demand<R, C>(
    client: &C,
    continuation: &CertifiedRootContinuation<R>,
    owners: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    metadata::tensor(
        client,
        owners,
        owners.numel(),
        DType::I64,
        "selected owners",
    )?;
    let state = &continuation.state;
    let original_count = state.status.numel();
    if owners.numel() > original_count {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Selected owner count {} exceeds original row count {original_count}. Supply unique in-range owners.",
                owners.numel()
            ),
        });
    }
    let dtype = state.lower.dtype();
    if owners.numel() == 0 {
        return Ok(Tensor::zeros(state.owner.shape(), dtype, client.device())?);
    }
    let zero = Tensor::zeros(&[], DType::I64, client.device())?;
    let limit = Tensor::full_scalar(&[], DType::I64, original_count as f64, client.device())?;
    let outside = client.maximum(&client.lt(owners, &zero)?, &client.ge(owners, &limit)?)?;
    if native::any(client, &client.cast(&outside, dtype)?)? {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "Selected owners fall outside [0, {original_count}). Supply original input-row indices."
            ),
        });
    }
    if owners.numel() > 1 {
        let sorted = client.sort(owners, 0, false)?;
        let left = sorted.narrow(0, 0, owners.numel() - 1)?;
        let right = sorted.narrow(0, 1, owners.numel() - 1)?;
        if native::any(client, &client.cast(&client.eq(&left, &right)?, dtype)?)? {
            return Err(OptimizeError::InvalidInput {
                context: "Selected owners contain duplicate indices. Supply each original input-row owner once.".into(),
            });
        }
    }
    let original = Tensor::zeros(&[original_count], dtype, client.device())?;
    let selected = Tensor::ones(owners.shape(), dtype, client.device())?;
    let original = client.scatter(&original, 0, owners, &selected)?;
    Ok(client.index_select(&original, 0, &state.owner)?)
}

/// Discard precision-policy history only for demanded nonsingleton compact roots.
///
/// Probe state stays. Its samples only steer probes, and each old nominal value stays
/// within its old enclosure width of the same residual.
pub(super) fn refresh<R, C>(
    client: &C,
    continuation: &mut CertifiedRootContinuation<R>,
    demand: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let state = &mut continuation.state;
    let nonsingleton = client.lt(&state.lower, &state.upper)?;
    let roots = native::indices(client, &client.mul(demand, &nonsingleton)?)?;
    if roots.numel() == 0 {
        return Ok(());
    }
    let zero = Tensor::zeros(roots.shape(), state.lower.dtype(), client.device())?;
    continuation.floor = native::replace(client, &continuation.floor, &roots, &zero)?;
    state.sampled = native::replace(client, &state.sampled, &roots, &zero)?;
    state.previous_point = native::replace(client, &state.previous_point, &roots, &zero)?;
    Ok(())
}
