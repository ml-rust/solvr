use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, ArithmeticResult, Interval};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

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

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    match dtype {
        DType::F32 => Tensor::from_slice(
            &values.iter().map(|&value| value as f32).collect::<Vec<_>>(),
            shape,
            device,
        ),
        DType::F64 => Tensor::from_slice(values, shape, device),
        _ => unreachable!("test inputs require F32 or F64"),
    }
}

pub fn read<R: Runtime<DType = DType>>(value: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
    match value.dtype() {
        DType::F32 => Ok(value
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        DType::F64 => value.try_to_vec::<f64>(),
        _ => unreachable!("test output requires F32 or F64"),
    }
}

pub fn input_bits<R, C>(client: &C, value: &Tensor<R>) -> numr::error::Result<Vec<u64>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), client.device())?;
    client.copy_into(&owned, value)?;
    match value.dtype() {
        DType::F32 => Ok(owned
            .try_to_vec::<f32>()?
            .into_iter()
            .map(|value| u64::from(value.to_bits()))
            .collect()),
        DType::F64 => Ok(owned
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect()),
        _ => unreachable!("test input requires F32 or F64"),
    }
}

pub fn enclosed<R: Runtime<DType = DType>>(
    value: &Interval<R>,
    lower: &[f64],
    upper: &[f64],
) -> Result {
    let actual_lower = read(value.lower())?;
    let actual_upper = read(value.upper())?;
    assert_eq!(actual_lower.len(), lower.len());
    assert_eq!(actual_upper.len(), upper.len());
    for (index, ((&lo, &hi), (&expected_lo, &expected_hi))) in actual_lower
        .iter()
        .zip(&actual_upper)
        .zip(lower.iter().zip(upper))
        .enumerate()
    {
        assert!(
            lo.is_finite() && hi.is_finite(),
            "lane{index}: nonfinite interval"
        );
        assert!(
            lo <= expected_lo && hi >= expected_hi,
            "lane{index}: [{lo},{hi}] excludes [{expected_lo},{expected_hi}]"
        );
        assert!(lo <= hi);
    }
    Ok(())
}

pub fn failure<T>(result: ArithmeticResult<T>) -> ArithmeticError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("invalid arithmetic returned a result"),
    }
}

pub fn range(error: ArithmeticError, expected: &'static str) {
    assert!(matches!(error, ArithmeticError::Range { stage } if stage == expected));
}

pub fn eta(dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    }
}
