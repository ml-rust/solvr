use super::support::{self, Client, Result, enclosed, failure, range, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

fn signed_corners<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    // Every corner product and quotient here is exactly binary representable.
    let a_lo = [0.0, -4.0, 0.0, -4.0, -4.0, -4.0, 2.0, -4.0];
    let a_hi = [4.0, 0.0, 4.0, 0.0, 4.0, 4.0, 4.0, -2.0];
    let b_lo = [2.0, -4.0, -4.0, 2.0, 2.0, -4.0, 2.0, -4.0];
    let b_hi = [4.0, 0.0, 0.0, 4.0, 4.0, -2.0, 4.0, -2.0];
    let a_lo = tensor::<R>(&a_lo, &[8], dtype, client.device())?;
    let a_hi = tensor::<R>(&a_hi, &[8], dtype, client.device())?;
    let b_lo = tensor::<R>(&b_lo, &[8], dtype, client.device())?;
    let b_hi = tensor::<R>(&b_hi, &[8], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let four = tensor::<R>(&[4.0], &[], dtype, client.device())?;
    let (product, point_division, interval_division) =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "signed exact corners", |context| {
            let a = context.bounds(&a_lo, &a_hi)?;
            let b = context.bounds(&b_lo, &b_hi)?;
            let divisor = context.bounds(&two, &four)?;
            Ok((
                context.mul(&a, &b)?,
                context.div_positive(&a, &two)?,
                context.div_positive_interval(&a, &divisor)?,
            ))
        })?;
    enclosed(
        &product,
        &[0.0, 0.0, -16.0, -16.0, -16.0, -16.0, 4.0, 4.0],
        &[16.0, 16.0, 0.0, 0.0, 16.0, 16.0, 16.0, 16.0],
    )?;
    let lower = support::read(product.lower())?;
    let upper = support::read(product.upper())?;
    for lane in [0, 1, 6, 7] {
        assert!(lower[lane] >= 0.0, "lane {lane}: {lower:?}");
    }
    for lane in [2, 3] {
        assert!(upper[lane] <= 0.0, "lane {lane}: {upper:?}");
    }
    for lane in [4, 5] {
        assert!(lower[lane] < 0.0 && upper[lane] > 0.0);
    }
    enclosed(
        &point_division,
        &[0.0, -2.0, 0.0, -2.0, -2.0, -2.0, 1.0, -2.0],
        &[2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, -1.0],
    )?;
    enclosed(
        &interval_division,
        &[0.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.5, -2.0],
        &[2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, -0.5],
    )?;
    for quotient in [&point_division, &interval_division] {
        let lower = support::read(quotient.lower())?;
        let upper = support::read(quotient.upper())?;
        for lane in [0, 2, 6] {
            assert!(lower[lane] >= 0.0);
        }
        for lane in [1, 3, 7] {
            assert!(upper[lane] <= 0.0);
        }
        for lane in [4, 5] {
            assert!(lower[lane] < 0.0 && upper[lane] > 0.0);
        }
    }
    Ok(())
}

fn nonnegative_chains<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = tensor::<R>(&[0.0, 0.0], &[2, 1], dtype, client.device())?;
    let upper = tensor::<R>(
        &[99.0, 4.0, 16.0, 99.0, 8.0, 32.0],
        &[2, 3],
        dtype,
        client.device(),
    )?
    .narrow(1, 1, 2)?
    .transpose(0, 1)?;
    let lower = lower.broadcast_to(&[2, 2])?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let four = tensor::<R>(&[4.0], &[], dtype, client.device())?;
    let half = tensor::<R>(&[0.5], &[], dtype, client.device())?;
    let results = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "nonnegative strided chains",
        |context| {
            let input = context.bounds(&lower, &upper)?;
            let divisor = context.bounds(&two, &four)?;
            let factor = context.bounds(&half, &two)?;
            let point_quotient = context.div_positive(&input, &two)?;
            let interval_quotient = context.div_positive_interval(&input, &divisor)?;
            let mut results = Vec::new();
            for quotient in [point_quotient, interval_quotient] {
                let product = context.mul(&quotient, &factor)?;
                let root = context.sqrt_nonnegative(&product)?;
                results.push((quotient, product, root));
            }
            Ok(results)
        },
    )?;
    for (quotient, product, root) in results {
        assert_eq!(root.lower().shape(), &[2, 2]);
        enclosed(&quotient, &[0.0; 4], &[2.0, 4.0, 8.0, 16.0])?;
        enclosed(&product, &[0.0; 4], &[4.0, 8.0, 16.0, 32.0])?;
        // Square roots at the first and third corners are exact integers.
        enclosed(&root, &[0.0; 4], &[2.0, 2.0, 4.0, 4.0])?;
        for value in [&quotient, &product, &root] {
            assert_eq!(support::read(value.lower())?, [0.0; 4]);
            assert!(support::read(value.upper())?.iter().all(|&x| x > 0.0));
        }
    }
    Ok(())
}

fn native_f64_chain<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let device = client.device();
    let zero = tensor::<R>(&[0.0], &[], DType::F64, device)?;
    let upper = tensor::<R>(
        &[f64::from_bits(0x3cf0000000000013)],
        &[],
        DType::F64,
        device,
    )?;
    let divisor = tensor::<R>(
        &[f64::from_bits(0x3b78f9be50d9e3c0)],
        &[],
        DType::F64,
        device,
    )?;
    let factor_lower = tensor::<R>(
        &[f64::from_bits(0x3b9ea053ef4dbf1c)],
        &[],
        DType::F64,
        device,
    )?;
    let factor_upper = tensor::<R>(
        &[f64::from_bits(0x3d139eb8708bd926)],
        &[],
        DType::F64,
        device,
    )?;
    let results = with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "native F64 nonnegative chain",
        |context| {
            let input = context.bounds(&zero, &upper)?;
            let denominator = context.point(&divisor)?;
            let factor = context.bounds(&factor_lower, &factor_upper)?;
            let mut results = Vec::new();
            for quotient in [
                context.div_positive(&input, &divisor)?,
                context.div_positive_interval(&input, &denominator)?,
            ] {
                let product = context.mul(&quotient, &factor)?;
                results.push(context.sqrt_nonnegative(&product)?);
                results.push(product);
                results.push(quotient);
            }
            Ok(results)
        },
    )?;
    // The represented inputs imply quotient > 2^23 and product > 2^-23.
    // Therefore each square-root upper endpoint also exceeds 2^-12.
    let corner_lower_bounds = [
        f64::from_bits(0x3f30000000000000),
        f64::from_bits(0x3e80000000000000),
        8_388_608.0,
    ];
    for (index, result) in results.into_iter().enumerate() {
        assert_eq!(support::read(result.lower())?, [0.0]);
        assert!(support::read(result.upper())?[0] >= corner_lower_bounds[index % 3]);
    }
    Ok(())
}

fn underflow_and_zero_bits<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let eta = support::eta(dtype);
    let tiny = tensor::<R>(&[eta, -eta], &[2], dtype, client.device())?;
    let half = tensor::<R>(&[0.5], &[], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    let zeros = tensor::<R>(&[0.0, -0.0], &[2], dtype, client.device())?;
    let (underflows, identities) = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "signed underflow and zeros",
        |context| {
            let tiny = context.point(&tiny)?;
            let half = context.point(&half)?;
            let denominator = context.point(&two)?;
            let zeros = context.point(&zeros)?;
            Ok((
                [
                    context.mul(&tiny, &half)?,
                    context.div_positive(&tiny, &two)?,
                    context.div_positive_interval(&tiny, &denominator)?,
                ],
                [
                    context.mul(&zeros, &denominator)?,
                    context.div_positive(&zeros, &two)?,
                    context.div_positive_interval(&zeros, &denominator)?,
                ],
            ))
        },
    )?;
    for result in underflows {
        // ±eta/2 lies strictly between zero and the adjacent subnormal.
        enclosed(&result, &[0.0, -eta], &[eta, 0.0])?;
        let lower = support::read(result.lower())?;
        let upper = support::read(result.upper())?;
        assert!(lower[0] >= 0.0 && upper[0] > 0.0);
        assert!(lower[1] < 0.0 && upper[1] <= 0.0);
    }
    let sign_bit = if dtype == DType::F32 {
        1 << 31
    } else {
        1 << 63
    };
    for result in identities {
        for endpoint in [result.lower(), result.upper()] {
            assert_eq!(support::input_bits(client, endpoint)?, [0, sign_bit]);
        }
    }
    Ok(())
}

fn rejected_inputs<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let zero = tensor::<R>(&[0.0], &[], dtype, client.device())?;
    let two = tensor::<R>(&[2.0], &[], dtype, client.device())?;
    for (lo, hi) in [
        (2.0, 0.0),
        (f64::NAN, 2.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ] {
        let lower = tensor::<R>(&[lo], &[], dtype, client.device())?;
        let upper = tensor::<R>(&[hi], &[], dtype, client.device())?;
        let error = failure(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "invalid signed operands",
            |context| {
                let input = context.bounds(&lower, &upper)?;
                let zero = context.point(&zero)?;
                let product = context.mul(&input, &zero)?;
                context.div_positive(&product, &two)
            },
        ));
        range(error, "invalid signed operands");
    }
    for denominator_lower in [0.0, -2.0] {
        let lower = tensor::<R>(&[denominator_lower], &[], dtype, client.device())?;
        let error = failure(with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "zero crossing denominator",
            |context| {
                let numerator = context.point(&zero)?;
                let denominator = context.bounds(&lower, &two)?;
                context.div_positive_interval(&numerator, &denominator)
            },
        ));
        range(error, "zero crossing denominator");
    }
    let error = failure(with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "zero exact denominator",
        |context| {
            let numerator = context.point(&zero)?;
            context.div_positive(&numerator, &zero)
        },
    ));
    range(error, "zero exact denominator");
    Ok(())
}

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for dtype in [DType::F32, DType::F64] {
        signed_corners::<R, _>(client, dtype)?;
        nonnegative_chains::<R, _>(client, dtype)?;
        underflow_and_zero_bits::<R, _>(client, dtype)?;
        rejected_inputs::<R, _>(client, dtype)?;
    }
    native_f64_chain::<R, _>(client)
}

#[test]
fn cpu_signed_intervals_preserve_enclosures_and_composable_domains() -> Result {
    let client = CpuClient::new(CpuDevice::new());
    run::<CpuRuntime, _>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_signed_intervals_preserve_enclosures_and_composable_domains() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)
}
