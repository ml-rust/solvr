use super::support::{Client, Result, failure, range, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

pub fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    for lower in [0.0, -1.0] {
        let lower = tensor::<R>(&[lower], &[1], dtype, client.device())?;
        let error = failure(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "invalid denominator",
            |context| {
                let numerator = context.point(&zero)?;
                let denominator = context.bounds(&lower, &two)?;
                context.div_positive_interval(&numerator, &denominator)
            },
        ));
        range(error, "invalid denominator");
    }

    let maximum = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let minimum_normal = if dtype == DType::F32 {
        f64::from(f32::MIN_POSITIVE)
    } else {
        f64::MIN_POSITIVE
    };
    let maximum = tensor::<R>(&[maximum], &[1], dtype, client.device())?;
    let minimum_normal = tensor::<R>(&[minimum_normal], &[1], dtype, client.device())?;
    let mut returned_finite = false;
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "overflow completion",
        |context| {
            let numerator = context.point(&maximum)?;
            let denominator = context.point(&minimum_normal)?;
            let _discarded = context.div_positive_interval(&numerator, &denominator)?;
            returned_finite = true;
            Ok(zero.clone())
        },
    ));
    assert!(returned_finite);
    range(error, "overflow completion");

    let mismatched = tensor::<R>(&[0.0, 1.0], &[2], dtype, client.device())?;
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "shape metadata",
        |context| context.bounds(&zero, &mismatched),
    ));
    assert!(matches!(error, ArithmeticError::Numr(_)));
    let other_dtype = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    let wrong_dtype = tensor::<R>(&[0.0], &[1], other_dtype, client.device())?;
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "dtype metadata",
        |context| context.point(&wrong_dtype),
    ));
    assert!(matches!(error, ArithmeticError::Numr(_)));

    for invalid in [f64::NAN, f64::INFINITY] {
        let invalid = tensor::<R>(&[invalid], &[1], dtype, client.device())?;
        let error = failure(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "nonfinite endpoints",
            |context| context.point(&invalid),
        ));
        range(error, "nonfinite endpoints");
    }
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "inverted endpoints",
        |context| context.bounds(&two, &zero),
    ));
    range(error, "inverted endpoints");

    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "closure error",
        |_context| -> std::result::Result<(), ArithmeticError> {
            Err(ArithmeticError::Range {
                stage: "closure sentinel",
            })
        },
    ));
    range(error, "closure sentinel");
    let mut called = false;
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::I64,
        "unsupported dtype",
        |_context| {
            called = true;
            Ok(())
        },
    ));
    assert!(!called);
    assert!(matches!(
        error,
        ArithmeticError::UnsupportedDType { dtype: DType::I64 }
    ));

    let negative = tensor::<R>(&[-super::support::eta(dtype)], &[1], dtype, client.device())?;
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "negative square root",
        |context| {
            let value = context.bounds(&negative, &zero)?;
            context.sqrt_nonnegative(&value)
        },
    ));
    range(error, "negative square root");
    Ok(())
}
