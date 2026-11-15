use super::support::{self, Client, Result, enclosed, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

pub fn division<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = tensor::<R>(&[-4.0, -1.0, 2.0, 0.0], &[4], dtype, client.device())?;
    let upper = tensor::<R>(&[-2.0, 3.0, 5.0, 0.0], &[4], dtype, client.device())?;
    let divisor_lower = tensor::<R>(&[0.5], &[1], dtype, client.device())?;
    let divisor_upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let original = support::input_bits(client, &lower)?;
    let result =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "signed division", |context| {
            let numerator = context.bounds(&lower, &upper)?;
            let denominator = context.bounds(&divisor_lower, &divisor_upper)?;
            context.div_positive_interval(&numerator, &denominator)
        })?;
    enclosed(&result, &[-8.0, -2.0, 1.0, 0.0], &[-1.0, 6.0, 10.0, 0.0])?;
    assert_eq!(support::read(result.lower())?[3], 0.0);
    assert_eq!(support::read(result.upper())?[3], 0.0);
    assert_eq!(support::input_bits(client, &lower)?, original);

    let numerator_lower = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let numerator_upper = tensor::<R>(&[5.0], &[1], dtype, client.device())?;
    let exact =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "exact divisor", |context| {
            let numerator = context.bounds(&numerator_lower, &numerator_upper)?;
            context.div_positive(&numerator, &divisor_upper)
        })?;
    enclosed(&exact, &[1.0], &[2.5])?;

    let tiny = tensor::<R>(&[support::eta(dtype)], &[1], dtype, client.device())?;
    let tiny_result =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "underflow division", |context| {
            let numerator = context.point(&tiny)?;
            let denominator = context.point(&divisor_upper)?;
            context.div_positive_interval(&numerator, &denominator)
        })?;
    enclosed(&tiny_result, &[0.0], &[support::eta(dtype)])?;
    assert!(support::read(tiny_result.upper())?[0] > 0.0);
    Ok(())
}

pub fn elementary<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = tensor::<R>(&[-2.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[3.0], &[1], dtype, client.device())?;
    let other_lower = tensor::<R>(&[-5.0], &[1], dtype, client.device())?;
    let other_upper = tensor::<R>(&[7.0], &[1], dtype, client.device())?;
    let (sum, difference, product) = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "elementary arithmetic",
        |context| {
            let left = context.bounds(&lower, &upper)?;
            let right = context.bounds(&other_lower, &other_upper)?;
            Ok((
                context.add(&left, &right)?,
                context.sub(&left, &right)?,
                context.mul(&left, &right)?,
            ))
        },
    )?;
    enclosed(&sum, &[-7.0], &[10.0])?;
    enclosed(&difference, &[-9.0], &[8.0])?;
    enclosed(&product, &[-15.0], &[21.0])
}

pub fn square_root<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let tiny = support::eta(dtype);
    let lower = tensor::<R>(&[0.0, 0.0, 2.0, tiny], &[4], dtype, client.device())?;
    let upper = tensor::<R>(&[0.0, 4.0, 9.0, tiny], &[4], dtype, client.device())?;
    let roots = with_interval_arithmetic::<R, _, _, _>(client, dtype, "square roots", |context| {
        let input = context.bounds(&lower, &upper)?;
        context.sqrt_nonnegative(&input)
    })?;
    let (sqrt_two_floor, sqrt_tiny_floor, sqrt_tiny_ceil) = if dtype == DType::F32 {
        (
            f64::from(f32::from_bits(0x3fb504f3)),
            f64::from(f32::from_bits(0x1a3504f3)),
            f64::from(f32::from_bits(0x1a3504f4)),
        )
    } else {
        (
            f64::from_bits(0x3ff6a09e667f3bcc),
            f64::from_bits(0x1e60000000000000),
            f64::from_bits(0x1e60000000000000),
        )
    };
    enclosed(
        &roots,
        &[0.0, 0.0, sqrt_two_floor, sqrt_tiny_floor],
        &[0.0, 2.0, 3.0, sqrt_tiny_ceil],
    )?;
    let lo = support::read(roots.lower())?;
    let hi = support::read(roots.upper())?;
    assert_eq!((lo[0], hi[0]), (0.0, 0.0));
    assert_eq!(lo[1], 0.0);
    let empty = Tensor::<R>::empty(&[2, 0, 3], dtype, client.device())?;
    let empty_roots =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "empty square roots", |context| {
            let value = context.point(&empty)?;
            context.sqrt_nonnegative(&value)
        })?;
    for endpoint in [empty_roots.lower(), empty_roots.upper()] {
        assert_eq!(endpoint.shape(), &[2, 0, 3]);
        assert_eq!(endpoint.dtype(), dtype);
        assert!(endpoint.device().is_same(client.device()));
        assert!(support::read(endpoint)?.is_empty());
    }
    Ok(())
}
