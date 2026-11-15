use super::support::{self, Client, Result, enclosed, failure, range, tensor};
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;
use solvr::common::arithmetic::{Interval, with_interval_arithmetic};

fn singleton<R: Runtime<DType = DType>>(value: &Interval<R>, expected: &[f64]) -> Result {
    for endpoint in [value.lower(), value.upper()] {
        let actual = support::read(endpoint)?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    Ok(())
}

fn epsilon(dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from(f32::EPSILON)
    } else {
        f64::EPSILON
    }
}

fn elementary<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let values = [3.0, 4.0, 7.0, 14.0, 2.0, -3.0];
    let inputs: Vec<_> = values
        .iter()
        .map(|&x| tensor::<R>(&[x], &[], dtype, client.device()))
        .collect::<numr::error::Result<_>>()?;
    let results = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "exact elementary results",
        |context| {
            let points: Vec<_> = inputs
                .iter()
                .map(|x| context.point(x))
                .collect::<std::result::Result<_, _>>()?;
            Ok(vec![
                context.add(&points[0], &points[1])?,
                context.sub(&points[2], &points[0])?,
                context.mul(&points[0], &points[1])?,
                context.div_positive(&points[3], &inputs[4])?,
                context.div_positive_interval(&points[3], &points[4])?,
                context.add(&points[0], &points[5])?,
                context.sub(&points[0], &points[0])?,
                context.sub(&points[0], &points[2])?,
                context.mul(&points[5], &points[1])?,
                context.div_positive(&points[5], &inputs[4])?,
                context.div_positive_interval(&points[5], &points[4])?,
            ])
        },
    )?;
    for (value, expected) in results
        .iter()
        .zip([7.0, 4.0, 12.0, 7.0, 7.0, 0.0, 0.0, -4.0, -12.0, -1.5, -1.5])
    {
        assert!(value.lower().shape().is_empty());
        singleton(value, &[expected])?;
    }
    Ok(())
}

fn nonexact_and_views<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let eps = epsilon(dtype);
    let a = tensor::<R>(
        &[0.0, 3.0, 1.0 + eps, 0.0, 5.0, 1.0 + eps],
        &[2, 3],
        dtype,
        client.device(),
    )?
    .narrow(1, 1, 2)?
    .transpose(0, 1)?;
    let factor = tensor::<R>(&[4.0, 1.0 + eps], &[2, 1], dtype, client.device())?;
    let half_eps = tensor::<R>(&[eps / 2.0], &[], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    let three = tensor::<R>(&[3.0], &[], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let (products, sum, quotient, wide) = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "nonexact native views",
        |context| {
            let left = context.point(&a)?;
            let right = context.point(&factor)?;
            let one_point = context.point(&one)?;
            let half_point = context.point(&half_eps)?;
            let three_point = context.point(&three)?;
            let nonsingleton = context.bounds(&two, &three)?;
            Ok((
                context.mul(&left, &right)?,
                context.add(&one_point, &half_point)?,
                context.div_positive_interval(&one_point, &three_point)?,
                vec![
                    context.add(&nonsingleton, &three_point)?,
                    context.sub(&nonsingleton, &three_point)?,
                    context.mul(&nonsingleton, &three_point)?,
                    context.div_positive_interval(&nonsingleton, &three_point)?,
                ],
            ))
        },
    )?;
    assert_eq!(products.lower().shape(), &[2, 2]);
    let lo = support::read(products.lower())?;
    let hi = support::read(products.upper())?;
    assert_eq!(&lo[..2], &[12.0, 20.0]);
    assert_eq!(&hi[..2], &[12.0, 20.0]);
    // (1 + eps)^2 lies strictly between these adjacent representable values.
    for index in 2..4 {
        assert!(lo[index] <= 1.0 + 2.0 * eps);
        assert!(hi[index] >= 1.0 + 3.0 * eps);
        assert!(lo[index] < hi[index]);
    }
    // The exact sum is halfway between one and its successor.
    enclosed(&sum, &[1.0], &[1.0 + eps])?;
    let (third_floor, third_ceil) = if dtype == DType::F32 {
        (
            f64::from(f32::from_bits(0x3eaaaaaa)),
            f64::from(f32::from_bits(0x3eaaaaab)),
        )
    } else {
        (
            f64::from_bits(0x3fd5555555555555),
            f64::from_bits(0x3fd5555555555556),
        )
    };
    enclosed(&quotient, &[third_floor], &[third_ceil])?;
    enclosed(&wide[0], &[5.0], &[6.0])?;
    enclosed(&wide[1], &[-1.0], &[0.0])?;
    enclosed(&wide[2], &[6.0], &[9.0])?;
    enclosed(&wide[3], &[2.0 * third_floor], &[1.0])?;
    for result in &wide {
        assert!(support::read(result.lower())?[0] < support::read(result.upper())?[0]);
    }
    Ok(())
}

fn residual_underflow<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (a_bits, b_bits, normal, subnormal, normal_successor, quotient_predecessor) =
        if dtype == DType::F32 {
            (
                f64::from(f32::from_bits(0x20000001)),
                f64::from(f32::from_bits(0x1ffffffe)),
                f64::from(f32::MIN_POSITIVE),
                f64::from(f32::from_bits(0x007fffff)),
                f64::from(f32::from_bits(0x00800001)),
                f64::from(f32::from_bits(0x007ffffe)),
            )
        } else {
            (
                f64::from_bits(0x2000000000000001),
                f64::from_bits(0x1ffffffffffffffe),
                f64::MIN_POSITIVE,
                f64::from_bits(0x000fffffffffffff),
                f64::from_bits(0x0010000000000001),
                f64::from_bits(0x000ffffffffffffe),
            )
        };
    let eta = support::eta(dtype);
    let a = tensor::<R>(
        &[a_bits, normal, eta, eta, 3.0],
        &[5],
        dtype,
        client.device(),
    )?;
    let b = tensor::<R>(
        &[b_bits, 1.0 + epsilon(dtype), 0.5, 2.0, 4.0],
        &[5],
        dtype,
        client.device(),
    )?;
    let (products, quotients, exact_divisors) =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "residual underflow", |context| {
            let left = context.point(&a)?;
            let right = context.point(&b)?;
            Ok((
                context.mul(&left, &right)?,
                context.div_positive_interval(&left, &right)?,
                context.div_positive(&left, &b)?,
            ))
        })?;
    let lo = support::read(products.lower())?;
    let hi = support::read(products.upper())?;
    // The product is min_normal minus 2^-1126 (F64) or 2^-172 (F32).
    assert_eq!((lo[0], hi[0]), (subnormal, normal_successor));
    assert!(lo[2] <= 0.0 && hi[2] >= eta);
    assert!(lo[2] < hi[2]);
    assert_eq!((lo[4], hi[4]), (12.0, 12.0));
    for result in [&quotients, &exact_divisors] {
        let lo = support::read(result.lower())?;
        let hi = support::read(result.upper())?;
        // The quotient exceeds max_subnormal despite a zero rounded FMA residual.
        assert_eq!((lo[1], hi[1]), (quotient_predecessor, normal));
        assert!(lo[3] <= 0.0 && hi[3] >= eta);
        assert!(lo[3] < hi[3]);
        assert_eq!((lo[4], hi[4]), (0.75, 0.75));
    }
    Ok(())
}

fn cap_recurrence<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = tensor::<R>(&[0.0], &[], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let three = tensor::<R>(&[3.0], &[], dtype, client.device())?;
    let six = tensor::<R>(&[6.0], &[], dtype, client.device())?;
    let slopes: Vec<_> = [2.0, 1.0, 3.0]
        .into_iter()
        .map(|x| tensor::<R>(&[x], &[], dtype, client.device()))
        .collect::<numr::error::Result<_>>()?;
    let (rectangle, area, width, integral) =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "piecewise linear cap", |context| {
            let mut area = context.point(&zero)?;
            let mut width = context.point(&two)?;
            let mut integral = context.point(&zero)?;
            let unit = context.point(&one)?;
            let depth = context.point(&three)?;
            let rectangle = context.mul(&width, &depth)?;
            for slope in &slopes {
                let slope = context.point(slope)?;
                let width_step = context.mul(&width, &unit)?;
                let slope_step = context.mul(&slope, &unit)?;
                let half_slope = context.div_positive(&slope_step, &two)?;
                let area_without_slope = context.add(&area, &width_step)?;
                let next_area = context.add(&area_without_slope, &half_slope)?;
                let area_step = context.mul(&area, &unit)?;
                let half_width = context.div_positive(&width_step, &two)?;
                let sixth_slope = context.div_positive(&slope_step, &six)?;
                let integral_area = context.add(&integral, &area_step)?;
                let integral_width = context.add(&integral_area, &half_width)?;
                integral = context.add(&integral_width, &sixth_slope)?;
                width = context.add(&width, &slope_step)?;
                area = next_area;
            }
            Ok((rectangle, area, width, integral))
        })?;
    singleton(&rectangle, &[6.0])?;
    singleton(&area, &[14.0])?;
    singleton(&width, &[8.0])?;
    enclosed(&integral, &[17.0], &[17.0])?;
    let integral_width = support::read(integral.upper())?[0] - support::read(integral.lower())?[0];
    assert!(integral_width <= 128.0 * epsilon(dtype) * 17.0);
    let above = if dtype == DType::F32 {
        f64::from(f32::from_bits(14.0f32.to_bits() + 1))
    } else {
        f64::from_bits(14.0f64.to_bits() + 1)
    };
    assert!(above > support::read(area.upper())?[0]);
    Ok(())
}

fn zero_identities_and_sticky_errors<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let negative_zero = tensor::<R>(&[-0.0], &[], dtype, client.device())?;
    let positive_zero = tensor::<R>(&[0.0], &[], dtype, client.device())?;
    let negative_two = tensor::<R>(&[-2.0], &[], dtype, client.device())?;
    let positive_two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let zeros = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "signed zero identities",
        |context| {
            let negative = context.point(&negative_zero)?;
            let positive = context.point(&positive_zero)?;
            let minus_two = context.point(&negative_two)?;
            let two = context.point(&positive_two)?;
            Ok(vec![
                context.add(&positive, &negative)?,
                context.mul(&negative, &minus_two)?,
                context.div_positive(&negative, &positive_two)?,
                context.div_positive_interval(&negative, &two)?,
                context.sub(&positive, &negative)?,
            ])
        },
    )?;
    let expected: u64 = if dtype == DType::F32 {
        0x80000000
    } else {
        0x8000000000000000
    };
    for (index, value) in zeros.into_iter().enumerate() {
        let bits = if index == 4 { 0 } else { expected };
        assert_eq!(support::input_bits(client, value.lower())?, [bits]);
        assert_eq!(support::input_bits(client, value.upper())?, [bits]);
    }
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "sticky invalid divisor",
        |context| {
            let zero = context.point(&positive_zero)?;
            let two = context.point(&positive_two)?;
            let _discarded = context.div_positive_interval(&two, &zero)?;
            context.add(&two, &two)
        },
    ));
    range(error, "sticky invalid divisor");
    Ok(())
}

fn guard_boundaries<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (minimum, maximum, outside_minimum, outside_maximum) = if dtype == DType::F32 {
        let lower_bits = 87u32 << 23;
        let upper_bits = 167u32 << 23;
        (
            f64::from(f32::from_bits(lower_bits)),
            f64::from(f32::from_bits(upper_bits)),
            f64::from(f32::from_bits(lower_bits - 1)),
            f64::from(f32::from_bits(upper_bits + 1)),
        )
    } else {
        let lower_bits = 573u64 << 52;
        let upper_bits = 1473u64 << 52;
        (
            f64::from_bits(lower_bits),
            f64::from_bits(upper_bits),
            f64::from_bits(lower_bits - 1),
            f64::from_bits(upper_bits + 1),
        )
    };
    let values = [minimum, maximum, outside_minimum, outside_maximum];
    let negative: Vec<_> = values.iter().map(|&x| -x).collect();
    let a = tensor::<R>(&values, &[4], dtype, client.device())?;
    let negative = tensor::<R>(&negative, &[4], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    let results = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "inclusive exactness guards",
        |context| {
            let a = context.point(&a)?;
            let negative = context.point(&negative)?;
            let unit = context.point(&one)?;
            Ok(vec![
                context.mul(&a, &unit)?,
                context.div_positive(&a, &one)?,
                context.div_positive_interval(&a, &unit)?,
                context.add(&a, &negative)?,
                context.sub(&a, &a)?,
            ])
        },
    )?;
    for (operation, result) in results.iter().enumerate() {
        let lower = support::read(result.lower())?;
        let upper = support::read(result.upper())?;
        for lane in 0..4 {
            let exact = if operation < 3 { values[lane] } else { 0.0 };
            assert!(lower[lane] <= exact && exact <= upper[lane]);
            if lane < 2 || operation < 3 || lane == 2 {
                assert_eq!(lower[lane].to_bits(), exact.to_bits());
                assert_eq!(upper[lane].to_bits(), exact.to_bits());
            } else {
                assert!(lower[lane] < upper[lane]);
            }
        }
    }
    Ok(())
}

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for dtype in [DType::F32, DType::F64] {
        elementary::<R, _>(client, dtype)?;
        guard_boundaries::<R, _>(client, dtype)?;
        nonexact_and_views::<R, _>(client, dtype)?;
        residual_underflow::<R, _>(client, dtype)?;
        cap_recurrence::<R, _>(client, dtype)?;
        zero_identities_and_sticky_errors::<R, _>(client, dtype)?;
    }
    Ok(())
}

#[test]
fn cpu_exact_singletons_preserve_independent_references_and_underflow_bounds() -> Result {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device).with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    run::<CpuRuntime, _>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_exact_singletons_preserve_independent_references_and_underflow_bounds() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)
}
