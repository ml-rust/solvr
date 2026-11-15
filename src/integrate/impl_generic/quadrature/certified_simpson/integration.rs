//! Coordinate the fixed certified integration stages.

use super::{admission, assemble, evaluate, queries};
use crate::DType;
use crate::common::arithmetic::interval::IntervalArithmetic;
use crate::integrate::error::IntegrateResult;
use crate::integrate::traits::{
    CertifiedSimpsonOptions, CertifiedSimpsonPanel, CertifiedSimpsonQuery, CertifiedSimpsonResult,
    CertifiedSimpsonValues,
};
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn certified_simpson_impl<R, C, F, D>(
    client: &C,
    left: &Tensor<R>,
    right: &Tensor<R>,
    options: CertifiedSimpsonOptions,
    values: F,
    fourth_bound: D,
) -> IntegrateResult<CertifiedSimpsonResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + TwoSumOps<R>
        + NextRepresentableOps<R>
        + RuntimeClient<R>,
    F: FnMut(&CertifiedSimpsonQuery<R>) -> IntegrateResult<CertifiedSimpsonValues<R>>,
    D: FnMut(&CertifiedSimpsonPanel<R>) -> IntegrateResult<Tensor<R>>,
{
    admission::metadata(client, left, right, options)?;
    if left.numel() == 0 {
        return assemble::zeros(client, 0, left.dtype());
    }
    admission::endpoints(client, left, right)?;
    let positive = client.lt(left, right)?;
    let indices = client.nonzero(&positive)?;
    let active = indices.numel();
    let original = indices.reshape(&[active])?;
    let plan = admission::plan(left.numel(), active, left.dtype(), options)?;
    if active == 0 {
        return assemble::zeros(client, left.numel(), left.dtype());
    }
    let original = client.sort(&original, 0, false)?;
    let mut arithmetic = IntervalArithmetic::new(client, left.dtype())?;
    let queries = queries::create(client, &mut arithmetic, left, right, original, plan)?;
    let evaluated = evaluate::callbacks(client, &queries, values, fourth_bound)?;
    assemble::finish(client, &mut arithmetic, queries, evaluated)
}
