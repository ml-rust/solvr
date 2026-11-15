//! Coordinate-independent stopping policy for the shared root iteration.

use super::{convergence, native::RootClient};
use crate::DType;
use crate::common::arithmetic::IntervalContext;
use crate::optimize::OptimizeResult;
use crate::optimize::roots::traits::{CertifiedBisectionOptions, CertifiedRootTolerance};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) trait Stopping<R, C>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    fn parameter_singleton_floor(&self) -> bool {
        false
    }

    fn satisfied(
        &mut self,
        ctx: &mut IntervalContext<'_, R, C>,
        owner: &Tensor<R>,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<Tensor<R>>;
}

pub(crate) struct Identity;

impl<R, C> Stopping<R, C> for Identity
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    fn satisfied(
        &mut self,
        ctx: &mut IntervalContext<'_, R, C>,
        _owner: &Tensor<R>,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        options: &CertifiedBisectionOptions,
    ) -> OptimizeResult<Tensor<R>> {
        mapped_width(ctx, lower, upper, options)
    }
}

pub(crate) fn mapped_width<R, C>(
    ctx: &mut IntervalContext<'_, R, C>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    options: &CertifiedBisectionOptions,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    convergence::satisfied(
        ctx,
        lower,
        upper,
        &CertifiedRootTolerance {
            atol: options.atol,
            rtol: options.rtol,
        },
    )
}
