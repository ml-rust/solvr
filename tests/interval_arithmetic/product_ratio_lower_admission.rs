use super::support::{self, Client, Result, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

pub(super) fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let eta = tensor::<R>(&[support::eta(dtype)], &[], dtype, client.device())?;
    let error = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "unrepresented positive ratio lower",
        |context| context.positive_product_ratio_lower(&[&eta], &[&two]),
    )
    .err()
    .ok_or("Accepted a positive quotient below minsub")?;
    assert!(matches!(error, ArithmeticError::Range { .. }));
    for invalid in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let scalar = tensor::<R>(&[invalid], &[], dtype, client.device())?;
        let mixed = tensor::<R>(&[1.0, invalid, 2.0], &[3], dtype, client.device())?;
        for factor in [&scalar, &mixed] {
            for numerator in [false, true] {
                let error = with_interval_arithmetic::<R, _, _, _>(
                    client,
                    dtype,
                    "invalid original ratio factor",
                    |context| {
                        if numerator {
                            context.positive_product_ratio_lower(&[&one, factor], &[&one])
                        } else {
                            context.positive_product_ratio_lower(&[&one], &[&one, factor])
                        }
                    },
                )
                .err()
                .ok_or("Accepted an invalid original positive factor")?;
                assert!(matches!(error, ArithmeticError::Range { .. }));
            }
        }
        for shape in [&[0][..], &[0, 2][..]] {
            let empty = Tensor::<R>::empty(shape, dtype, client.device())?;
            for numerator in [false, true] {
                let error = with_interval_arithmetic::<R, _, _, _>(
                    client,
                    dtype,
                    "invalid scalar before empty broadcast",
                    |context| {
                        if numerator {
                            context.positive_product_ratio_lower(&[&empty, &scalar], &[&one])
                        } else {
                            context.positive_product_ratio_lower(&[&empty], &[&scalar])
                        }
                    },
                )
                .err()
                .ok_or("Empty output concealed an invalid scalar factor")?;
                assert!(matches!(error, ArithmeticError::Range { .. }));
            }
        }
    }
    let other_dtype = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    let wrong_dtype = tensor::<R>(&[1.0], &[], other_dtype, client.device())?;
    for numerator in [false, true] {
        let error = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "ratio factor dtype",
            |context| {
                if numerator {
                    context.positive_product_ratio_lower(&[&wrong_dtype], &[&one])
                } else {
                    context.positive_product_ratio_lower(&[&one], &[&wrong_dtype])
                }
            },
        )
        .err()
        .ok_or("Accepted a mismatched factor dtype")?;
        assert!(matches!(error, ArithmeticError::Numr(_)));
    }
    let a = tensor::<R>(&[1.0; 2], &[2], dtype, client.device())?;
    let b = tensor::<R>(&[1.0; 3], &[3], dtype, client.device())?;
    let error = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "incompatible ratio factor shapes",
        |context| context.positive_product_ratio_lower(&[&a], &[&b]),
    )
    .err()
    .ok_or("Accepted incompatible factor shapes")?;
    assert!(matches!(error, ArithmeticError::Numr(_)));
    Ok(())
}
