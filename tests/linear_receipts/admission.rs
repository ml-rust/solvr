use super::cpu::setup;
use super::fixture::{Result, tensor};
use numr::prelude::*;
use solvr::interpolate::{InterpolateError, LinearInterp1dDiagnostics};

#[test]
fn receipt_metadata_rejects_rank_lengths_and_unsupported_or_mixed_dtypes() -> Result {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let make = |values: &[f64]| tensor(values, &[values.len()], dtype, &device);
        let x = make(&[0.0, 1.0])?;
        let y = make(&[2.0, 3.0])?;
        let q = make(&[0.5])?;
        let matrix = x.reshape(&[1, 2])?;
        let scalar = q.reshape(&[])?;
        for (axis, ordinate, query) in [(&matrix, &y, &q), (&x, &matrix, &q), (&x, &y, &scalar)] {
            assert!(matches!(
                client.interp1d_linear_with_receipts(axis, ordinate, query),
                Err(InterpolateError::DimensionMismatch { .. })
            ));
        }
        let short = y.narrow(0, 0, 1)?;
        assert!(matches!(
            client.interp1d_linear_with_receipts(&x, &short, &q),
            Err(InterpolateError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            client.interp1d_linear_with_receipts(&short, &short, &q),
            Err(InterpolateError::InsufficientData { .. })
        ));
        let empty = Tensor::<CpuRuntime>::empty(&[0], dtype, &device)?;
        assert!(matches!(
            client.interp1d_linear_with_receipts(&empty, &empty, &q),
            Err(InterpolateError::InsufficientData { .. })
        ));
        let other = tensor(
            &[0.0, 1.0],
            &[2],
            if dtype == DType::F32 {
                DType::F64
            } else {
                DType::F32
            },
            &device,
        )?;
        assert!(matches!(
            client.interp1d_linear_with_receipts(&x, &other, &q),
            Err(InterpolateError::InvalidParameter { .. })
        ));
        assert!(matches!(
            client.interp1d_linear_with_receipts(&x, &y, &other),
            Err(InterpolateError::InvalidParameter { .. })
        ));
    }
    let integer = Tensor::from_slice(&[0i32, 1], &[2], &device)?;
    assert!(matches!(
        client.interp1d_linear_with_receipts(&integer, &integer, &integer),
        Err(InterpolateError::InvalidParameter { .. })
    ));
    Ok(())
}

#[test]
fn empty_queries_still_check_finite_data_monotonicity_and_grid_width() -> Result {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let make = |values: &[f64]| tensor(values, &[values.len()], dtype, &device);
        let x = make(&[0.0, 1.0])?;
        let y = make(&[0.0, 1.0])?;
        let empty = Tensor::<CpuRuntime>::empty(&[0], dtype, &device)?;
        for axis in [[0.0, 0.0], [1.0, 0.0]] {
            assert!(matches!(
                client.interp1d_linear_with_receipts(&make(&axis)?, &y, &empty),
                Err(InterpolateError::NotMonotonic { .. })
            ));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (axis, ordinate, query) in [
                (make(&[0.0, value])?, y.clone(), empty.clone()),
                (x.clone(), make(&[0.0, value])?, empty.clone()),
                (x.clone(), y.clone(), make(&[value])?),
            ] {
                let error = client
                    .interp1d_linear_with_receipts(&axis, &ordinate, &query)
                    .err()
                    .ok_or("Nonfinite receipt input accepted")?;
                assert!(!matches!(
                    error,
                    InterpolateError::UnsupportedArithmetic { .. }
                ));
            }
        }
        let maximum = if dtype == DType::F32 {
            f64::from(f32::MAX)
        } else {
            f64::MAX
        };
        assert!(matches!(
            client.interp1d_linear_with_receipts(&make(&[-maximum, maximum])?, &y, &empty),
            Err(InterpolateError::NumericalError { .. })
        ));
    }
    Ok(())
}

#[test]
fn closed_support_rejects_extrapolation_even_for_constant_ordinates() -> Result {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let make = |values: &[f64]| tensor(values, &[values.len()], dtype, &device);
        for values in [[0.0, 1.0], [7.0, 7.0]] {
            for query in [-1.0, 2.0] {
                assert!(
                    matches!(client.interp1d_linear_with_receipts(&make(&[0.0,1.0])?,&make(&values)?,&make(&[query])?),Err(InterpolateError::OutOfDomain { point,min,max,.. }) if point==query && min==0.0 && max==1.0)
                );
            }
        }
    }
    Ok(())
}
