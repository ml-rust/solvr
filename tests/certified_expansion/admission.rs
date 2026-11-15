use super::fixture::{Client, Result, rejected, tensor, values};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

pub fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let pair = tensor::<R>(&[1.0, 2.0], &[2], dtype, client.device())?;
    for position in [0, 1, 2] {
        let mut components = [&zero, &zero, &zero];
        components[position] = &pair;
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "component shape",
            |context| context.expansion_bounds(components[0], components[1], components[2]),
        ))?;
        assert!(matches!(error, ArithmeticError::Numr(_)));
    }
    for operation in 0..4 {
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "binary shape",
            |context| {
                let left = context.expansion_point(&one)?;
                let right = context.expansion_point(&pair)?;
                match operation {
                    0 => context.expansion_add(&left, &right),
                    1 => context.expansion_sub(&left, &right),
                    2 => context.expansion_mul(&left, &right),
                    _ => context.expansion_div_positive(&left, &right),
                }
            },
        ))?;
        assert!(matches!(error, ArithmeticError::Numr(_)));
    }
    let other_dtype = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    let wrong = tensor::<R>(&[0.0], &[1], other_dtype, client.device())?;
    for position in 0..3 {
        let mut components = [&zero, &zero, &zero];
        components[position] = &wrong;
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "component dtype",
            |context| context.expansion_bounds(components[0], components[1], components[2]),
        ))?;
        assert!(matches!(error, ArithmeticError::Numr(_)));
    }
    let error = rejected(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "point dtype",
        |context| context.expansion_point(&wrong),
    ))?;
    assert!(matches!(error, ArithmeticError::Numr(_)));
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = tensor::<R>(&[invalid], &[1], dtype, client.device())?;
        let mut returned_partial = false;
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "sticky expansion point",
            |context| {
                let _discarded = context.expansion_point(&invalid)?;
                returned_partial = true;
                context.expansion_point(&one)
            },
        ))?;
        assert!(returned_partial);
        assert!(matches!(
            error,
            ArithmeticError::Range {
                stage: "sticky expansion point"
            }
        ));
    }
    for (position, invalid) in [
        (0, f64::NAN),
        (0, f64::INFINITY),
        (1, f64::NEG_INFINITY),
        (2, f64::NAN),
        (2, f64::INFINITY),
        (2, -1.0),
    ] {
        let invalid = tensor::<R>(&[invalid], &[1], dtype, client.device())?;
        let mut components = [&zero, &zero, &zero];
        components[position] = &invalid;
        let mut returned_partial = false;
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "sticky expansion input",
            |context| {
                let _discarded =
                    context.expansion_bounds(components[0], components[1], components[2])?;
                returned_partial = true;
                context.expansion_point(&one)
            },
        ))?;
        assert!(returned_partial);
        assert!(matches!(
            error,
            ArithmeticError::Range {
                stage: "sticky expansion input"
            }
        ));
    }
    for (high, low, radius) in [
        (0.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ] {
        let high = tensor::<R>(&[high], &[1], dtype, client.device())?;
        let low = tensor::<R>(&[low], &[1], dtype, client.device())?;
        let radius = tensor::<R>(&[radius], &[1], dtype, client.device())?;
        let mut returned_partial = false;
        let error = rejected(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "expanded divisor",
            |context| {
                let numerator = context.expansion_point(&one)?;
                let denominator = context.expansion_bounds(&high, &low, &radius)?;
                let _discarded = context.expansion_div_positive(&numerator, &denominator)?;
                returned_partial = true;
                Ok(numerator)
            },
        ))?;
        assert!(returned_partial);
        assert!(matches!(
            error,
            ArithmeticError::Range {
                stage: "expanded divisor"
            }
        ));
    }
    shapes::<R, C>(client, dtype)
}

fn shapes<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for shape in [&[][..], &[2, 1, 3][..], &[2, 0, 3][..]] {
        let count = shape.iter().product::<usize>();
        let source = tensor::<R>(&vec![2.0; count], shape, dtype, client.device())?;
        let (sum, difference, product, quotient) =
            with_interval_arithmetic::<R, _, _, _>(client, dtype, "expansion ranks", |context| {
                let value = context.expansion_point(&source)?;
                Ok((
                    context.expansion_add(&value, &value)?,
                    context.expansion_sub(&value, &value)?,
                    context.expansion_mul(&value, &value)?,
                    context.expansion_div_positive(&value, &value)?,
                ))
            })?;
        for (result, expected) in [
            (&sum, 4.0),
            (&difference, 0.0),
            (&product, 4.0),
            (&quotient, 1.0),
        ] {
            for component in [result.high(), result.low(), result.abs_error()] {
                assert_eq!(component.shape(), shape);
                assert_eq!(component.dtype(), dtype);
                assert!(component.device().is_same(client.device()));
                assert!(component.is_contiguous());
            }
            assert_eq!(values(client, result.high())?, vec![expected; count]);
            assert_eq!(values(client, result.low())?, vec![0.0; count]);
            assert_eq!(values(client, result.abs_error())?, vec![0.0; count]);
        }
    }
    Ok(())
}
