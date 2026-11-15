//! Snapshot and check caller-provided mapped domain enclosures.

use super::super::certified::{
    metadata,
    native::{self, RootClient},
};
use super::super::certified_bisection::stopping::{self, Stopping};
use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedRootDomainQuery, CertifiedRootMappedValues,
};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct Mapped<'a, M> {
    pub(super) callback: &'a M,
    pub(super) points: usize,
}

impl<R, C, M> Stopping<R, C> for Mapped<'_, M>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    M: Fn(&CertifiedRootDomainQuery<R>) -> numr::error::Result<CertifiedRootMappedValues<R>>,
{
    fn parameter_singleton_floor(&self) -> bool {
        true
    }

    fn satisfied(
        &mut self,
        ctx: &mut IntervalContext<'_, R, C>,
        owner: &Tensor<R>,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<Tensor<R>> {
        let client = ctx.client();
        if owner.numel() == 0 {
            return Ok(Tensor::empty(
                lower.shape(),
                lower.dtype(),
                client.device(),
            )?);
        }
        let next = self.points.checked_add(owner.numel()).ok_or_else(|| {
            OptimizeError::NumericalError {
                message:
                    "Mapped root domain-point count overflows usize. Stop the continuation session."
                        .into(),
            }
        })?;
        let query = CertifiedRootDomainQuery {
            owner: native::owned(client, owner)?,
            lower: native::owned(client, lower)?,
            upper: native::owned(client, upper)?,
        };
        let values = (self.callback)(&query)?;
        self.points = next;
        for (name, value) in [
            ("mapping lower", &values.lower),
            ("mapping upper", &values.upper),
        ] {
            metadata::tensor(client, value, lower.numel(), lower.dtype(), name)?;
        }
        let low = native::owned(client, &values.lower)?;
        let high = native::owned(client, &values.upper)?;
        let invalid = client.maximum(
            &native::nonfinite(client, &low)?,
            &native::nonfinite(client, &high)?,
        )?;
        let invalid = client.maximum(&invalid, &client.gt(&low, &high)?)?;
        if native::any(client, &invalid)? {
            return Err(OptimizeError::InvalidInput {
                context: "Mapped root callback contains nonfinite or inverted bounds. Return finite ordered complete-domain bounds.".into(),
            });
        }
        stopping::mapped_width(ctx, &low, &high, options)
    }
}
