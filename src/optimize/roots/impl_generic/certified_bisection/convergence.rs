//! Compare rigorous physical width with an inward tolerance.

use super::native::{self, RootClient};
use super::{admission, scalar_tolerance};
use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::roots::traits::CertifiedRootTolerance;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn admitted_satisfied<R, C>(
    ctx: &mut IntervalContext<'_, R, C>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let client = ctx.client();
    admission::tensor(client, lower, lower.numel(), ctx.dtype(), "tolerance lower")?;
    admission::tensor(client, upper, lower.numel(), ctx.dtype(), "tolerance upper")?;
    admission::count(lower.numel(), ctx.dtype())?;
    admission::tolerances(tolerance.atol, tolerance.rtol)?;
    let invalid = client.maximum(
        &native::nonfinite(client, lower)?,
        &native::nonfinite(client, upper)?,
    )?;
    let invalid = client.maximum(&invalid, &client.gt(lower, upper)?)?;
    if native::any(client, &invalid)? {
        return Err(OptimizeError::InvalidInput {
            context: "Certified root tolerance bounds are nonfinite or inverted. Supply finite ordered complete bounds.".into(),
        });
    }
    satisfied(ctx, lower, upper, tolerance)
}

pub(crate) fn satisfied<R, C>(
    ctx: &mut IntervalContext<'_, R, C>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let client = ctx.client();
    let exact = client.eq(lower, upper)?;
    if tolerance.atol == 0.0 && tolerance.rtol == 0.0 {
        return Ok(exact);
    }
    let pending = native::indices(
        client,
        &client.eq(
            &exact,
            &Tensor::zeros(exact.shape(), lower.dtype(), client.device())?,
        )?,
    )?;
    if pending.numel() == 0 {
        return Ok(exact);
    }
    let lower = client.index_select(lower, 0, &pending)?;
    let upper = client.index_select(upper, 0, &pending)?;
    let dtype = lower.dtype();
    let atol = scalar_tolerance::lower(dtype, tolerance.atol)?;
    let rtol = scalar_tolerance::lower(dtype, tolerance.rtol)?;
    let a = Tensor::full_scalar(&[1], dtype, atol, client.device())?;
    let r = Tensor::full_scalar(&[1], dtype, rtol, client.device())?;
    let magnitude = client.maximum(&client.abs(&lower)?, &client.abs(&upper)?)?;
    let low = ctx.point(&lower).map_err(native::arithmetic)?;
    let high = ctx.point(&upper).map_err(native::arithmetic)?;
    let width = ctx.sub(&high, &low).map_err(native::arithmetic)?;
    let scale = ctx.point(&magnitude).map_err(native::arithmetic)?;
    let relative = ctx.point(&r).map_err(native::arithmetic)?;
    let relative = ctx.mul(&relative, &scale).map_err(native::arithmetic)?;
    let absolute = ctx.point(&a).map_err(native::arithmetic)?;
    let tolerance = ctx.add(&absolute, &relative).map_err(native::arithmetic)?;
    let accepted = client.le(width.upper(), tolerance.lower())?;
    native::replace(client, &exact, &pending, &accepted)
}
