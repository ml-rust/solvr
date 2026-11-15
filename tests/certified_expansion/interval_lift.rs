use super::fixture::{self, Client, Result, enclosed, input, tensor, values};
use numr::ops::MidpointOps;
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

fn points_and_ownership<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let maximum = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let minimum = if dtype == DType::F32 {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    for strided in [false, true] {
        let lower = input::<R>(
            &[-0.0, -0.0, 0.0, 1.25, minimum, maximum],
            dtype,
            client.device(),
            strided,
        )?;
        let upper = input::<R>(
            &[-0.0, 0.0, -0.0, 1.25, minimum, maximum],
            dtype,
            client.device(),
            strided,
        )?;
        let expected = fixture::bits(client, &lower)?;
        let (interval, lifted) =
            with_interval_arithmetic::<R, _, _, _>(client, dtype, "point interval lift", |a| {
                assert!(std::ptr::eq(a.client(), client));
                let interval = a.bounds(&lower, &upper)?;
                let lifted = a.expansion_from_interval(&interval)?;
                Ok((interval, lifted))
            })?;
        assert_eq!(fixture::bits(client, lifted.high())?, expected);
        assert!(
            values(client, lifted.low())?
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            values(client, lifted.abs_error())?
                .iter()
                .all(|&value| value == 0.0)
        );
        let before = [
            fixture::bits(client, lifted.high())?,
            fixture::bits(client, lifted.low())?,
            fixture::bits(client, lifted.abs_error())?,
        ];
        for component in [lifted.high(), lifted.low(), lifted.abs_error()] {
            assert_eq!(component.shape(), &[6]);
            assert_eq!(component.dtype(), dtype);
            assert!(component.device().is_same(client.device()));
            assert!(component.is_contiguous());
            for source in [&lower, &upper, interval.lower(), interval.upper()] {
                assert!(!component.shares_storage_with(source));
            }
        }
        let replacement = tensor::<R>(&[17.0; 6], &[6], dtype, client.device())?;
        for source in [&lower, &upper, interval.lower(), interval.upper()] {
            client.copy_into(source, &replacement)?;
        }
        for (component, expected) in [lifted.high(), lifted.low(), lifted.abs_error()]
            .into_iter()
            .zip(before)
        {
            assert_eq!(fixture::bits(client, component)?, expected);
        }
    }
    Ok(())
}

fn ranges<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let (minimum, one_up) = if dtype == DType::F32 {
        (
            f64::from(f32::from_bits(1)),
            f64::from(f32::from_bits(0x3f800001)),
        )
    } else {
        (f64::from_bits(1), f64::from_bits(0x3ff0000000000001))
    };
    let lower_values = [0.0, minimum, -minimum, 1.0];
    let upper_values = [minimum, 2.0 * minimum, minimum, one_up];
    for strided in [false, true] {
        let lower = input::<R>(&lower_values, dtype, client.device(), strided)?;
        let upper = input::<R>(&upper_values, dtype, client.device(), strided)?;
        let (lifted, projected) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "subnormal interval lift",
            |a| {
                let interval = a.bounds(&lower, &upper)?;
                let lifted = a.expansion_from_interval(&interval)?;
                let projected = a.expansion_interval(&lifted)?;
                Ok((lifted, projected))
            },
        )?;
        let actual_lower = values(client, projected.lower())?;
        let actual_upper = values(client, projected.upper())?;
        let radii = values(client, lifted.abs_error())?;
        for index in 0..lower_values.len() {
            assert!(actual_lower[index] <= lower_values[index]);
            assert!(actual_upper[index] >= upper_values[index]);
            assert!(radii[index].is_finite() && radii[index] > 0.0);
        }
    }
    let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
    let lifted =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "empty interval lift", |a| {
            let interval = a.bounds(&empty, &empty)?;
            a.expansion_from_interval(&interval)
        })?;
    for component in [lifted.high(), lifted.low(), lifted.abs_error()] {
        assert_eq!(component.shape(), &[0]);
        assert_eq!(component.dtype(), dtype);
        assert!(component.device().is_same(client.device()));
    }
    Ok(())
}

fn extremes<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let (maximum, required_radius) = if dtype == DType::F32 {
        (f64::from(f32::MAX), f64::from(f32::from_bits(0x7f400000)))
    } else {
        (f64::MAX, f64::from_bits(0x7fe8000000000000))
    };
    let lower = tensor::<R>(&[-maximum], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[maximum], &[1], dtype, client.device())?;
    let (lifted, projected) =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "opposite finite extremes", |a| {
            let interval = a.bounds(&lower, &upper)?;
            let lifted = a.expansion_from_interval(&interval)?;
            let projected = a.expansion_interval(&lifted)?;
            Ok((lifted, projected))
        })?;
    assert_eq!(values(client, lifted.high())?, [0.0]);
    assert_eq!(values(client, lifted.abs_error())?, [maximum]);
    enclosed(
        client,
        &projected,
        [-maximum, maximum],
        "opposite finite extremes",
    )?;

    let upper = tensor::<R>(&[maximum * 0.5], &[1], dtype, client.device())?;
    let lifted =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "finite ball overhang", |a| {
            let interval = a.bounds(&lower, &upper)?;
            a.expansion_from_interval(&interval)
        })?;
    for component in [lifted.high(), lifted.low(), lifted.abs_error()] {
        assert!(
            values(client, component)?
                .iter()
                .all(|value| value.is_finite())
        );
    }
    assert!(values(client, lifted.abs_error())?[0] >= required_radius);
    let error = fixture::rejected(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "overhanging ball projection",
        |a| a.expansion_interval(&lifted),
    ))?;
    assert!(matches!(error, ArithmeticError::Range { .. }));
    Ok(())
}

fn cancellation<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let (tiny, tiny_upper, relative) = if dtype == DType::F32 {
        (
            f64::from(f32::from_bits(0x2b800000)),
            f64::from(f32::from_bits(0x2b800001)),
            1e-5,
        )
    } else {
        (
            f64::from_bits(0x39b0000000000000),
            f64::from_bits(0x39b0000000000001),
            1e-10,
        )
    };
    for strided in [false, true] {
        let lower = input::<R>(&[tiny; 2], dtype, client.device(), strided)?;
        let upper = input::<R>(&[tiny, tiny_upper], dtype, client.device(), strided)?;
        let negative_two = input::<R>(&[-2.0; 2], dtype, client.device(), strided)?;
        let positive_two = input::<R>(&[2.0; 2], dtype, client.device(), strided)?;
        let (retained, late) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "lift before cancellation",
            |a| {
                let interval = a.bounds(&lower, &upper)?;
                let lifted = a.expansion_from_interval(&interval)?;
                let negative = a.expansion_point(&negative_two)?;
                let positive = a.expansion_point(&positive_two)?;
                let combined = a.expansion_add(&negative, &lifted)?;
                let cancelled = a.expansion_add(&combined, &positive)?;
                let retained = a.expansion_interval(&cancelled)?;

                let negative = a.point(&negative_two)?;
                let rounded = a.add(&negative, &interval)?;
                let lifted_late = a.expansion_from_interval(&rounded)?;
                let cancelled_late = a.expansion_add(&lifted_late, &positive)?;
                Ok((retained, a.expansion_interval(&cancelled_late)?))
            },
        )?;
        let actual_lower = values(client, retained.lower())?;
        let actual_upper = values(client, retained.upper())?;
        for (index, expected_upper) in [tiny, tiny_upper].into_iter().enumerate() {
            assert!(0.0 < actual_lower[index] && actual_lower[index] <= tiny);
            assert!(expected_upper <= actual_upper[index]);
            let error = (actual_lower[index] - expected_upper)
                .abs()
                .max((actual_upper[index] - tiny).abs());
            assert!(
                error <= relative * tiny,
                "retained tiny interval: error={error}"
            );
        }
        assert!(
            values(client, late.lower())?
                .iter()
                .all(|&value| value <= 0.0)
        );
    }
    Ok(())
}

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    for dtype in [DType::F32, DType::F64] {
        points_and_ownership::<R, _>(client, dtype)?;
        ranges::<R, _>(client, dtype)?;
        extremes::<R, _>(client, dtype)?;
        cancellation::<R, _>(client, dtype)?;
    }
    Ok(())
}

#[test]
fn configured_cpu_interval_lifts_retain_uncertainty_and_cancellation() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let device = CpuDevice::new();
    let base = CpuRuntime::default_client(&device);
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    run::<CpuRuntime, _>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_interval_lifts_retain_uncertainty_and_cancellation() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_interval_lifts_reject_unqualified_arithmetic() -> Result {
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        let empty = Tensor::<WgpuRuntime>::empty(&[0], dtype, client.device())?;
        let mut called = false;
        let error = fixture::rejected(with_interval_arithmetic::<WgpuRuntime, _, _, _>(
            &client,
            dtype,
            "unqualified interval lift",
            |a| {
                called = true;
                let interval = a.bounds(&empty, &empty)?;
                a.expansion_from_interval(&interval)
            },
        ))?;
        assert!(!called);
        assert!(
            matches!(error, ArithmeticError::UnsupportedArithmetic { dtype: actual, .. } if actual == dtype)
        );
    }
    Ok(())
}
