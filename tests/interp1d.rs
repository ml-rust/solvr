//! Checked common-grid linear interpolation on a configured CPU client.

#[path = "interp1d/cases.rs"]
mod cases;

use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};
use solvr::interpolate::{InterpolateError, InterpolateResult};

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (client, device)
}

#[test]
fn linear_preserves_tiny_span_knots_midpoints_extrapolation_and_layouts() -> InterpolateResult<()> {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        cases::observations(&client, &device, dtype)?;
    }
    Ok(())
}

#[test]
fn linear_retains_minimum_subnormal_width_without_a_denominator_floor() -> InterpolateResult<()> {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let tiny = if dtype == DType::F32 {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        let x = cases::tensor(&[0.0, tiny], &[2], dtype, &device)?;
        let y = cases::tensor(&[0.0, 1.0], &[2], dtype, &device)?;
        let q = cases::tensor(&[0.0, tiny, 2.0 * tiny], &[3], dtype, &device)?;
        assert_eq!(
            cases::values(&client.interp1d(&x, &y, &q, InterpMethod::Linear)?)?,
            vec![0.0, 1.0, 2.0]
        );
    }
    Ok(())
}

#[test]
fn exact_knots_preserve_stored_bits_and_skip_unneeded_overflow() -> InterpolateResult<()> {
    let (client, device) = setup();
    let x = Tensor::from_slice(&[0.0f64, 1.0, 2.0, 3.0], &[4], &device)?;
    let ordinates = [f64::MAX, -f64::MAX, -0.0, 7.0];
    let y = Tensor::from_slice(&ordinates, &[4], &device)?;
    let output = client.interp1d(&x, &y, &x, InterpMethod::Linear)?;
    assert_eq!(
        output
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        ordinates.map(f64::to_bits)
    );
    let midpoint = Tensor::from_slice(&[0.5f64], &[1], &device)?;
    assert!(matches!(
        client.interp1d(&x, &y, &midpoint, InterpMethod::Linear),
        Err(InterpolateError::NumericalError { .. })
    ));
    Ok(())
}

#[test]
fn active_unrepresentable_linear_intermediates_return_explicit_errors() -> InterpolateResult<()> {
    let (client, device) = setup();
    let tiny = f64::from_bits(1);
    for (axis, ordinates, query) in [
        ([0.0, 1.0], [f64::MAX, -f64::MAX], 0.5),
        ([0.0, tiny], [0.0, 1.0], f64::MAX),
        ([-f64::MAX, -f64::MAX / 2.0], [0.0, 1.0], f64::MAX),
        ([0.0, 1.0], [0.0, tiny], tiny),
    ] {
        let x = Tensor::from_slice(&axis, &[2], &device)?;
        let y = Tensor::from_slice(&ordinates, &[2], &device)?;
        let q = Tensor::from_slice(&[query], &[1], &device)?;
        assert!(matches!(
            client.interp1d(&x, &y, &q, InterpMethod::Linear),
            Err(InterpolateError::NumericalError { .. })
        ));
    }
    Ok(())
}

#[test]
fn shared_metadata_checks_reject_rank_length_and_insufficient_grid() -> InterpolateResult<()> {
    let (client, device) = setup();
    let x = Tensor::from_slice(&[0.0f64, 1.0], &[2], &device)?;
    let y = Tensor::from_slice(&[0.0f64, 2.0], &[2], &device)?;
    let q = Tensor::from_slice(&[0.5f64], &[1], &device)?;
    let matrix = x.reshape(&[1, 2])?;
    let scalar = q.reshape(&[])?;
    let short = y.narrow(0, 0, 1)?;
    let empty = Tensor::<CpuRuntime>::empty(&[0], DType::F64, &device)?;
    for method in [
        InterpMethod::Linear,
        InterpMethod::Nearest,
        InterpMethod::Cubic,
    ] {
        for (axis, ordinate, query) in [(&matrix, &y, &q), (&x, &matrix, &q), (&x, &y, &scalar)] {
            assert!(matches!(
                client.interp1d(axis, ordinate, query, method),
                Err(InterpolateError::DimensionMismatch { .. })
            ));
        }
        assert!(matches!(
            client.interp1d(&x, &short, &q, method),
            Err(InterpolateError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            client.interp1d(&short, &short, &q, method),
            Err(InterpolateError::InsufficientData { .. })
        ));
        assert!(matches!(
            client.interp1d(&empty, &empty, &q, method),
            Err(InterpolateError::InsufficientData { .. })
        ));
    }
    Ok(())
}

#[test]
fn shared_checks_reject_duplicate_reversed_and_nonfinite_grid_or_queries() -> InterpolateResult<()>
{
    let (client, device) = setup();
    let make = |values: &[f64]| Tensor::from_slice(values, &[values.len()], &device);
    let x = make(&[0.0, 1.0])?;
    let y = make(&[0.0, 1.0])?;
    let q = make(&[0.5])?;
    let empty = Tensor::<CpuRuntime>::empty(&[0], DType::F64, &device)?;
    for axis in [[0.0, 0.0], [1.0, 0.0]] {
        assert!(matches!(
            client.interp1d(&make(&axis)?, &y, &q, InterpMethod::Linear),
            Err(InterpolateError::NotMonotonic { .. })
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            client
                .interp1d(&make(&[0.0, value])?, &y, &q, InterpMethod::Linear)
                .is_err()
        );
        assert!(
            client
                .interp1d(&x, &make(&[0.0, value])?, &q, InterpMethod::Linear)
                .is_err()
        );
        assert!(
            client
                .interp1d(&x, &y, &make(&[value])?, InterpMethod::Linear)
                .is_err()
        );
        assert!(
            client
                .interp1d(&make(&[0.0, value])?, &y, &empty, InterpMethod::Linear)
                .is_err()
        );
    }
    assert!(
        client
            .interp1d(
                &make(&[-f64::MAX, f64::MAX])?,
                &y,
                &empty,
                InterpMethod::Linear
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn shared_dtype_checks_reject_mixed_or_nonfloating_inputs() -> InterpolateResult<()> {
    let (client, device) = setup();
    let x = Tensor::from_slice(&[0.0f64, 1.0], &[2], &device)?;
    let f32 = Tensor::from_slice(&[0.0f32, 1.0], &[2], &device)?;
    assert!(client.interp1d(&x, &f32, &x, InterpMethod::Linear).is_err());
    assert!(client.interp1d(&x, &x, &f32, InterpMethod::Linear).is_err());
    let integer = Tensor::from_slice(&[0i32, 1], &[2], &device)?;
    assert!(
        client
            .interp1d(&integer, &integer, &integer, InterpMethod::Linear)
            .is_err()
    );
    Ok(())
}

#[test]
fn constant_intervals_skip_extreme_coordinate_arithmetic_in_mixed_lanes() -> InterpolateResult<()> {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let (tiny, large) = if dtype == DType::F32 {
            (f64::from(f32::from_bits(1)), f64::from(f32::MAX))
        } else {
            (f64::from_bits(1), f64::MAX)
        };
        let make = |data: &[f64]| cases::tensor(data, &[data.len()], dtype, &device);
        let constant = make(&[7.0, 7.0])?;
        for (axis, query) in [([0.0, tiny], [1.0]), ([0.0, large], [tiny])] {
            let output = client.interp1d(
                &make(&axis)?,
                &constant,
                &make(&query)?,
                InterpMethod::Linear,
            )?;
            assert_eq!(cases::values(&output)?, vec![7.0]);
        }
        let output = client.interp1d(
            &make(&[0.0, tiny, 1.0])?,
            &make(&[7.0, 7.0, 9.0])?,
            &make(&[-1.0, 0.5, 1.0])?,
            InterpMethod::Linear,
        )?;
        assert_eq!(cases::values(&output)?, vec![7.0, 8.0, 9.0]);
        let output = client.interp1d(
            &make(&[-1.0, 0.0, large])?,
            &make(&[5.0, 7.0, 7.0])?,
            &make(&[tiny, -0.5, large])?,
            InterpMethod::Linear,
        )?;
        assert_eq!(cases::values(&output)?, vec![7.0, 6.0, 7.0]);
    }
    Ok(())
}
