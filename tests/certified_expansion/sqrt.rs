use super::fixture::{self, Client, Result, enclosed, input, tensor, values};
use numr::ops::MidpointOps;
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

fn exact_roots<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    for strided in [false, true] {
        let source = input::<R>(
            &[-0.0, 0.0, 0.25, 1.0, 4.0, 9.0, 16.0, 256.0],
            dtype,
            client.device(),
            strided,
        )?;
        let source_bits = fixture::bits(client, &source)?;
        // Existing complete-input projection determines the output zero signs.
        let expected = tensor::<R>(
            &[0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 16.0],
            &[8],
            dtype,
            client.device(),
        )?;
        let (captured, root) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "exact expansion square roots",
            |a| {
                assert!(std::ptr::eq(a.client(), client));
                let captured = a.expansion_point(&source)?;
                let root = a.expansion_sqrt_nonnegative(&captured)?;
                Ok((captured, root))
            },
        )?;
        assert_eq!(fixture::bits(client, &source)?, source_bits);
        assert_eq!(fixture::bits(client, captured.high())?, source_bits);
        assert_eq!(
            fixture::bits(client, root.high())?,
            fixture::bits(client, &expected)?
        );
        assert!(
            values(client, root.low())?
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            values(client, root.abs_error())?
                .iter()
                .all(|&value| value == 0.0)
        );
        let before = [
            fixture::bits(client, root.high())?,
            fixture::bits(client, root.low())?,
            fixture::bits(client, root.abs_error())?,
        ];
        for component in [root.high(), root.low(), root.abs_error()] {
            assert_eq!(component.shape(), &[8]);
            assert_eq!(component.dtype(), dtype);
            assert!(component.device().is_same(client.device()));
            assert!(component.is_contiguous());
            for operand in [
                &source,
                captured.high(),
                captured.low(),
                captured.abs_error(),
            ] {
                assert!(!component.shares_storage_with(operand));
            }
        }
        let replacement = tensor::<R>(&[17.0; 8], &[8], dtype, client.device())?;
        for operand in [
            &source,
            captured.high(),
            captured.low(),
            captured.abs_error(),
        ] {
            client.copy_into(operand, &replacement)?;
        }
        for (component, expected) in [root.high(), root.low(), root.abs_error()]
            .into_iter()
            .zip(before)
        {
            assert_eq!(fixture::bits(client, component)?, expected);
        }
    }
    Ok(())
}

fn complete_inputs<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let (delta, nonsquare_low, one_up, sqrt_two) = if dtype == DType::F32 {
        (
            2.0_f64.powi(-13),
            2.0_f64.powi(-30),
            f64::from(f32::from_bits(0x3f800001)),
            [
                f64::from(f32::from_bits(0x3fb504f3)),
                f64::from(f32::from_bits(0x3fb504f4)),
            ],
        )
    } else {
        (
            2.0_f64.powi(-27),
            2.0_f64.powi(-80),
            f64::from_bits(0x3ff0000000000001),
            [
                f64::from_bits(0x3ff6a09e667f3bcc),
                f64::from_bits(0x3ff6a09e667f3bcd),
            ],
        )
    };
    for strided in [false, true] {
        let high = input::<R>(
            &[1.0 + 2.0 * delta, 1.0, 5.0, 2.0],
            dtype,
            client.device(),
            strided,
        )?;
        let low = input::<R>(
            &[delta * delta, nonsquare_low, 0.0, 0.0],
            dtype,
            client.device(),
            strided,
        )?;
        let radius = input::<R>(&[0.0, 0.0, 4.0, 0.0], dtype, client.device(), strided)?;
        let (source, root, projected) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "complete expansion square roots",
            |a| {
                let source = a.expansion_bounds(&high, &low, &radius)?;
                let root = a.expansion_sqrt_nonnegative(&source)?;
                let projected = a.expansion_interval(&root)?;
                Ok((source, root, projected))
            },
        )?;
        assert!(values(client, source.low())?[0] > 0.0);
        let centers = values(client, root.high())?;
        let radii = values(client, root.abs_error())?;
        assert_eq!(centers[0], 1.0 + delta);
        assert_eq!(values(client, root.low())?[0], 0.0);
        assert_eq!(radii[0], 0.0);
        let lower = values(client, projected.lower())?;
        let upper = values(client, projected.upper())?;
        for radius in &radii[1..4] {
            assert!(radius.is_finite() && *radius > 0.0);
        }
        assert!(lower[1] <= 1.0 && upper[1] >= one_up);
        assert!(lower[2] <= 1.0 && upper[2] >= 3.0);
        assert!(lower[3] <= sqrt_two[0] && upper[3] >= sqrt_two[1]);
    }
    Ok(())
}

fn ranges_and_empty<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    let (subnormal_square, exact_root, maximum, maximum_root) = if dtype == DType::F32 {
        (
            f64::from(f32::from_bits(2)),
            2.0_f64.powi(-74),
            f64::from(f32::MAX),
            [
                f64::from(f32::from_bits(0x5f7fffff)),
                f64::from(f32::from_bits(0x5f800000)),
            ],
        )
    } else {
        (
            f64::from_bits(1),
            2.0_f64.powi(-537),
            f64::MAX,
            [
                f64::from_bits(0x5fefffffffffffff),
                f64::from_bits(0x5ff0000000000000),
            ],
        )
    };
    for (value, expected) in [(subnormal_square, [exact_root; 2]), (maximum, maximum_root)] {
        let value = tensor::<R>(&[value], &[1], dtype, client.device())?;
        let (root, projected) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "range-safe square proof",
            |a| {
                let source = a.expansion_point(&value)?;
                let root = a.expansion_sqrt_nonnegative(&source)?;
                let projected = a.expansion_interval(&root)?;
                Ok((root, projected))
            },
        )?;
        enclosed(client, &projected, expected, "range-safe square root")?;
        assert!(values(client, root.high())?[0].is_finite());
        assert!(values(client, root.abs_error())?[0] > 0.0);
    }
    let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
    let root = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "empty expansion square root",
        |a| {
            let source = a.expansion_point(&empty)?;
            a.expansion_sqrt_nonnegative(&source)
        },
    )?;
    for component in [root.high(), root.low(), root.abs_error()] {
        assert_eq!(component.shape(), &[0]);
        assert_eq!(component.dtype(), dtype);
    }
    let negative = tensor::<R>(&[-1.0], &[1], dtype, client.device())?;
    let error = fixture::rejected(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "negative expansion square root",
        |a| {
            let source = a.expansion_point(&negative)?;
            a.expansion_sqrt_nonnegative(&source)
        },
    ))?;
    assert!(matches!(error, ArithmeticError::Range { .. }));
    Ok(())
}

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R>,
{
    for dtype in [DType::F32, DType::F64] {
        exact_roots::<R, _>(client, dtype)?;
        complete_inputs::<R, _>(client, dtype)?;
        ranges_and_empty::<R, _>(client, dtype)?;
    }
    Ok(())
}

#[test]
fn configured_cpu_expansion_square_roots_preserve_complete_certificates() -> Result {
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
fn actual_cuda_expansion_square_roots_preserve_complete_certificates() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_expansion_square_roots_reject_unqualified_arithmetic() -> Result {
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        let empty = Tensor::<WgpuRuntime>::empty(&[0], dtype, client.device())?;
        let mut called = false;
        let error = fixture::rejected(with_interval_arithmetic::<WgpuRuntime, _, _, _>(
            &client,
            dtype,
            "unqualified expansion square root",
            |a| {
                called = true;
                let source = a.expansion_point(&empty)?;
                a.expansion_sqrt_nonnegative(&source)
            },
        ))?;
        assert!(!called);
        assert!(
            matches!(error, ArithmeticError::UnsupportedArithmetic { dtype: actual, .. } if actual == dtype)
        );
    }
    Ok(())
}
