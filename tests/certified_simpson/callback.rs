use super::fixture::{Case, Function, decode, range};
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps};
use numr::prelude::*;
use solvr::integrate::{CertifiedSimpsonQuery, CertifiedSimpsonValues, IntegrateResult};

pub trait Client<R: Runtime<DType = DType>>:
    TensorOps<R> + ScalarOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>
{
}
impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
}

struct Interval<R: Runtime<DType = DType>> {
    lower: Tensor<R>,
    upper: Tensor<R>,
}
fn constant<R, C>(client: &C, like: &Tensor<R>, value: f64) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    Ok(Tensor::full_scalar(
        like.shape(),
        like.dtype(),
        value,
        client.device(),
    )?)
}
fn point<R: Runtime<DType = DType>>(value: &Tensor<R>) -> Interval<R> {
    Interval {
        lower: value.clone(),
        upper: value.clone(),
    }
}
fn zero_mask<R, C>(client: &C, value: &Interval<R>) -> IntegrateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = constant(client, &value.lower, 0.0)?;
    Ok(client.mul(
        &client.eq(&value.lower, &zero)?,
        &client.eq(&value.upper, &zero)?,
    )?)
}
fn add<R, C>(client: &C, a: &Interval<R>, b: &Interval<R>) -> IntegrateResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let az = zero_mask(client, a)?;
    let bz = zero_mask(client, b)?;
    let lower = client.next_down(&client.add(&a.lower, &b.lower)?)?;
    let upper = client.next_up(&client.add(&a.upper, &b.upper)?)?;
    Ok(Interval {
        lower: client.where_cond(&az, &b.lower, &client.where_cond(&bz, &a.lower, &lower)?)?,
        upper: client.where_cond(&az, &b.upper, &client.where_cond(&bz, &a.upper, &upper)?)?,
    })
}
fn sub<R, C>(client: &C, a: &Interval<R>, b: &Interval<R>) -> IntegrateResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    add(
        client,
        a,
        &Interval {
            lower: client.neg(&b.upper)?,
            upper: client.neg(&b.lower)?,
        },
    )
}
fn mul<R, C>(client: &C, a: &Interval<R>, b: &Interval<R>) -> IntegrateResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = constant(client, &a.lower, 0.0)?;
    let exact_zero = client.maximum(&zero_mask(client, a)?, &zero_mask(client, b)?)?;
    let mut lower = constant(client, &a.lower, f64::INFINITY)?;
    let mut upper = constant(client, &a.lower, f64::NEG_INFINITY)?;
    for left in [&a.lower, &a.upper] {
        for right in [&b.lower, &b.upper] {
            let product = client.mul(left, right)?;
            lower = client.minimum(&lower, &client.next_down(&product)?)?;
            upper = client.maximum(&upper, &client.next_up(&product)?)?;
        }
    }
    Ok(Interval {
        lower: client.where_cond(&exact_zero, &zero, &lower)?,
        upper: client.where_cond(&exact_zero, &zero, &upper)?,
    })
}
pub fn any<R, C>(client: &C, mask: &Tensor<R>) -> IntegrateResult<bool>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let floating_mask = client.cast(mask, DType::F32)?;
    Ok(client.any(&floating_mask, &[], false)?.item::<f32>()? != 0.0)
}
fn finite<R, C>(client: &C, value: &Tensor<R>) -> IntegrateResult<bool>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    Ok(!any(
        client,
        &client.maximum(&client.isnan(value)?, &client.isinf(value)?)?,
    )?)
}
fn sqrt_interval<R, C>(client: &C, input: &Interval<R>) -> IntegrateResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = constant(client, &input.lower, 0.0)?;
    let lo_candidate = client.sqrt(&input.lower)?;
    let hi_candidate = client.sqrt(&input.upper)?;
    let lower = client.maximum(&zero, &client.next_down(&client.next_down(&lo_candidate)?)?)?;
    let upper = client.next_up(&client.next_up(&hi_candidate)?)?;
    let square_upper = client.next_up(&client.mul(&lower, &lower)?)?;
    let square_upper = client.where_cond(&client.eq(&lower, &zero)?, &zero, &square_upper)?;
    let square_lower = client.next_down(&client.mul(&upper, &upper)?)?;
    let invalid = client.maximum(
        &client.gt(&square_upper, &input.lower)?,
        &client.lt(&square_lower, &input.upper)?,
    )?;
    if !finite(client, &lower)? || !finite(client, &upper)? || any(client, &invalid)? {
        return Err(range("callback square certificate"));
    }
    Ok(Interval { lower, upper })
}
pub fn evaluate<R, C>(
    client: &C,
    case: &Case,
    query: &CertifiedSimpsonQuery<R>,
) -> IntegrateResult<CertifiedSimpsonValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let domain = Interval {
        lower: query.lower.clone(),
        upper: query.upper.clone(),
    };
    let (value, interval) = match case.function {
        Function::Polynomial {
            coefficients,
            scale,
            shift,
            offset,
        } => {
            let shift = constant(client, &query.point, decode(shift, case.dtype))?;
            let scale = constant(client, &query.point, decode(scale, case.dtype))?;
            let offset = constant(client, &query.point, decode(offset, case.dtype))?;
            let input = sub(client, &domain, &point(&shift))?;
            let nominal_input = client.sub(&query.point, &shift)?;
            let mut nominal = constant(client, &query.point, 0.0)?;
            let mut result = point(&nominal);
            for &coefficient in coefficients.iter().rev() {
                let coefficient = constant(client, &query.point, decode(coefficient, case.dtype))?;
                nominal = client.add(&client.mul(&nominal, &nominal_input)?, &coefficient)?;
                result = add(client, &mul(client, &result, &input)?, &point(&coefficient))?;
            }
            nominal = client.add(&client.mul(&scale, &nominal)?, &offset)?;
            result = add(
                client,
                &mul(client, &point(&scale), &result)?,
                &point(&offset),
            )?;
            (nominal, result)
        }
        Function::Sqrt => {
            let one = constant(client, &query.point, 1.0)?;
            let two = constant(client, &query.point, 2.0)?;
            let input = add(client, &point(&one), &domain)?;
            let input = Interval {
                lower: client.maximum(&one, &input.lower)?,
                upper: client.minimum(&two, &input.upper)?,
            };
            let nominal = client.sqrt(&client.add(&one, &query.point)?)?;
            (nominal, sqrt_interval(client, &input)?)
        }
    };
    let lower = client.minimum(&interval.lower, &value)?;
    let upper = client.maximum(&interval.upper, &value)?;
    if !finite(client, &value)? || !finite(client, &lower)? || !finite(client, &upper)? {
        return Err(range("callback function arithmetic"));
    }
    Ok(CertifiedSimpsonValues {
        value,
        lower,
        upper,
    })
}
