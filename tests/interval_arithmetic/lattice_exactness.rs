use super::support::{self, Client, Result, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

fn products<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let eta = support::eta(dtype);
    let left = tensor::<R>(
        &[eta, 5.0 * eta, -eta, eta, eta, 2.0 * eta],
        &[6],
        dtype,
        client.device(),
    )?;
    let right = tensor::<R>(
        &[5.0, 5.0, 5.0, 1.5, 0.5, 1.5],
        &[6],
        dtype,
        client.device(),
    )?;
    for swapped in [false, true] {
        let (interval, expansion) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "integer lattice products",
            |context| {
                let (left, right) = if swapped {
                    (&right, &left)
                } else {
                    (&left, &right)
                };
                let a = context.point(left)?;
                let b = context.point(right)?;
                let ea = context.expansion_point(left)?;
                let eb = context.expansion_point(right)?;
                Ok((context.mul(&a, &b)?, context.expansion_mul(&ea, &eb)?))
            },
        )?;
        let lower = support::read(interval.lower())?;
        let upper = support::read(interval.upper())?;
        let high = support::read(expansion.high())?;
        let low = support::read(expansion.low())?;
        let radius = support::read(expansion.abs_error())?;
        for (lane, expected) in [5.0 * eta, 25.0 * eta, -5.0 * eta].into_iter().enumerate() {
            assert_eq!((lower[lane], upper[lane]), (expected, expected));
            assert_eq!(high[lane] + low[lane], expected);
            assert_eq!(radius[lane], 0.0);
        }
        // Fractional factors can hide a sub-eta residual behind a rounded zero FMA.
        for lane in [3, 4] {
            assert!(lower[lane] < upper[lane]);
            assert!(radius[lane] > 0.0);
        }
        assert!(lower[3] <= eta && upper[3] >= 2.0 * eta);
        assert!(lower[4] <= 0.0 && upper[4] >= eta);
        assert!(lower[5] <= 3.0 * eta && upper[5] >= 3.0 * eta);
    }
    // An uncertain factor remains uncertain even when its center product is exact.
    let radius = tensor::<R>(&[eta; 6], &[6], dtype, client.device())?;
    let zero = tensor::<R>(&[0.0; 6], &[6], dtype, client.device())?;
    let result = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "uncertain lattice product",
        |context| {
            let a = context.expansion_bounds(&left, &zero, &radius)?;
            let b = context.expansion_point(&right)?;
            context.expansion_mul(&a, &b)
        },
    )?;
    assert!(support::read(result.abs_error())?[0] > 0.0);
    Ok(())
}

fn quotient_advection<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let eta = support::eta(dtype);
    let discharge = tensor::<R>(&[5.0 * eta, -5.0 * eta], &[2], dtype, client.device())?;
    let area = tensor::<R>(&[eta; 2], &[2], dtype, client.device())?;
    let (velocity, advection, velocity_interval, advection_interval) =
        with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "exact minimum-subnormal quotient and advection",
            |context| {
                let discharge = context.expansion_point(&discharge)?;
                let area = context.expansion_point(&area)?;
                let velocity = context.expansion_div_positive(&discharge, &area)?;
                let advection = context.expansion_mul(&discharge, &velocity)?;
                let velocity_interval = context.expansion_interval(&velocity)?;
                let advection_interval = context.expansion_interval(&advection)?;
                Ok((velocity, advection, velocity_interval, advection_interval))
            },
        )?;
    for (label, expansion, interval, expected) in [
        ("Q/A", &velocity, &velocity_interval, [5.0, -5.0]),
        ("Q*(Q/A)", &advection, &advection_interval, [25.0 * eta; 2]),
    ] {
        assert_eq!(
            support::read(expansion.high())?,
            expected,
            "{label}: exact high components for {dtype:?}"
        );
        assert_eq!(
            support::read(expansion.low())?,
            [0.0; 2],
            "{label}: exact low components for {dtype:?}"
        );
        assert_eq!(
            support::read(expansion.abs_error())?,
            [0.0; 2],
            "{label}: complete expansion radius for {dtype:?}"
        );
        assert_eq!(
            support::read(interval.lower())?,
            expected,
            "{label}: exact interval lower endpoints for {dtype:?}"
        );
        assert_eq!(
            support::read(interval.upper())?,
            expected,
            "{label}: exact interval upper endpoints for {dtype:?}"
        );
    }
    Ok(())
}

fn sums<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let eta = support::eta(dtype);
    let a_lo = tensor::<R>(&[25.0 * eta, -eta, 2.0, 1.0], &[4], dtype, client.device())?;
    let a_hi = tensor::<R>(&[25.0 * eta, eta, 3.0, 2.0], &[4], dtype, client.device())?;
    let b_lo = tensor::<R>(&[0.0, -eta, 3.0, 0.0], &[4], dtype, client.device())?;
    let half_epsilon = if dtype == DType::F32 {
        f64::from(f32::EPSILON) / 2.0
    } else {
        f64::EPSILON / 2.0
    };
    let b_hi = tensor::<R>(&[eta, eta, 4.0, half_epsilon], &[4], dtype, client.device())?;
    let (sum, difference) = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "independent sum endpoints",
        |context| {
            let a = context.bounds(&a_lo, &a_hi)?;
            let b = context.bounds(&b_lo, &b_hi)?;
            Ok((context.add(&a, &b)?, context.sub(&a, &b)?))
        },
    )?;
    let sum_lo = support::read(sum.lower())?;
    let sum_hi = support::read(sum.upper())?;
    let sub_lo = support::read(difference.lower())?;
    let sub_hi = support::read(difference.upper())?;
    for (lane, (lo, hi)) in [
        (25.0 * eta, 26.0 * eta),
        (-2.0 * eta, 2.0 * eta),
        (5.0, 7.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!((sum_lo[lane], sum_hi[lane]), (lo, hi));
    }
    for (lane, (lo, hi)) in [
        (24.0 * eta, 25.0 * eta),
        (-2.0 * eta, 2.0 * eta),
        (-2.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!((sub_lo[lane], sub_hi[lane]), (lo, hi));
    }
    // The lower sum endpoint is exact. Its inexact upper endpoint stays outward.
    assert_eq!(sum_lo[3], 1.0);
    assert!(sum_hi[3] > 2.0);
    assert_eq!(sub_hi[3], 2.0);
    assert!(sub_lo[3] <= 1.0 - half_epsilon);
    Ok(())
}

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for dtype in [DType::F32, DType::F64] {
        products::<R, _>(client, dtype)?;
        quotient_advection::<R, _>(client, dtype)?;
        sums::<R, _>(client, dtype)?;
    }
    Ok(())
}

#[test]
fn configured_cpu_lattice_products_and_sum_endpoints() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    run::<CpuRuntime, _>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_lattice_products_and_sum_endpoints() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)
}
