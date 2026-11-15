use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, ArithmeticResult, Interval};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

pub struct Case {
    pub id: &'static str,
    pub op: Operation,
    pub a: [u64; 3],
    pub b: [u64; 3],
    pub anchor: u64,
    pub residual: [u64; 2],
    pub max_abs_error: u64,
    pub zero_error: bool,
    pub nonzero_error: bool,
}

pub trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + ScalarOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + TwoSumOps<R>
    + RuntimeClient<R>
{
}

impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
}

pub fn decode(bits: u64, dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    }
}

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    if dtype == DType::F32 {
        Tensor::from_slice(
            &values.iter().map(|&value| value as f32).collect::<Vec<_>>(),
            shape,
            device,
        )
    } else {
        Tensor::from_slice(values, shape, device)
    }
}

pub fn input<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
    strided: bool,
) -> numr::error::Result<Tensor<R>> {
    if strided {
        let paired: Vec<_> = values.iter().flat_map(|&value| [value, 91.0]).collect();
        Ok(tensor::<R>(&paired, &[values.len(), 2], dtype, device)?
            .narrow(1, 0, 1)?
            .squeeze(Some(1)))
    } else {
        tensor(values, &[values.len()], dtype, device)
    }
}

pub fn bits<R, C>(client: &C, value: &Tensor<R>) -> numr::error::Result<Vec<u64>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), client.device())?;
    client.copy_into(&owned, value)?;
    if value.dtype() == DType::F32 {
        Ok(owned
            .try_to_vec::<f32>()?
            .into_iter()
            .map(|value| u64::from(value.to_bits()))
            .collect())
    } else {
        Ok(owned
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }
}

pub fn enclosed<R, C>(client: &C, value: &Interval<R>, expected: [f64; 2], id: &str) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = values(client, value.lower())?;
    let upper = values(client, value.upper())?;
    assert!(!lower.is_empty(), "{id}: empty reference result");
    assert_eq!(lower.len(), upper.len(), "{id}: endpoint shapes differ");
    for (&lo, &hi) in lower.iter().zip(&upper) {
        assert!(lo.is_finite() && hi.is_finite(), "{id}: [{lo}, {hi}]");
        assert!(
            lo <= expected[0] && hi >= expected[1],
            "{id}: [{lo}, {hi}] excludes {expected:?}"
        );
        assert!(lo <= hi, "{id}: inverted interval");
    }
    Ok(())
}

pub fn values<R, C>(client: &C, value: &Tensor<R>) -> numr::error::Result<Vec<f64>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let dtype = value.dtype();
    Ok(bits(client, value)?
        .into_iter()
        .map(|value| decode(value, dtype))
        .collect())
}

pub fn rejected<T>(result: ArithmeticResult<T>) -> Result<ArithmeticError> {
    match result {
        Err(error) => Ok(error),
        Ok(_) => Err("invalid expansion returned a result".into()),
    }
}
