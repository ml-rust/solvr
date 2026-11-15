use super::support::{self, Client, Result, failure, input_bits, range, read, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let maximum = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let tiny = support::eta(dtype);
    let lower = [
        tiny, -tiny, maximum, -maximum, -0.0, 0.0, -tiny, -maximum, -0.0,
    ];
    let upper = [
        tiny, -tiny, maximum, -maximum, -0.0, 0.0, tiny, maximum, 0.0,
    ];
    for strided in [false, true] {
        let component = |values: &[f64]| -> numr::error::Result<Tensor<R>> {
            if strided {
                let storage: Vec<_> = values.iter().flat_map(|&value| [91.0, value]).collect();
                Ok(
                    tensor::<R>(&storage, &[values.len(), 2], dtype, client.device())?
                        .narrow(1, 1, 1)?
                        .squeeze(Some(1)),
                )
            } else {
                tensor::<R>(values, &[values.len()], dtype, client.device())
            }
        };
        let lo = component(&lower)?;
        let hi = component(&upper)?;
        let denominator = component(&[1.0; 9])?;
        if strided {
            assert!(!lo.is_contiguous());
        }
        let expected = [input_bits(client, &lo)?, input_bits(client, &hi)?];
        let (numerator, divisor, exact, interval) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "complete exact-one division identities",
            |context| {
                let numerator = context.bounds(&lo, &hi)?;
                let divisor = context.point(&denominator)?;
                let exact = context.div_positive(&numerator, &denominator)?;
                let interval = context.div_positive_interval(&numerator, &divisor)?;
                Ok((numerator, divisor, exact, interval))
            },
        )?;
        for result in [&exact, &interval] {
            assert_eq!(input_bits(client, result.lower())?, expected[0]);
            assert_eq!(input_bits(client, result.upper())?, expected[1]);
            for endpoint in [result.lower(), result.upper()] {
                assert_eq!(endpoint.shape(), &[9]);
                assert_eq!(endpoint.dtype(), dtype);
                assert!(endpoint.device().is_same(client.device()));
                assert!(endpoint.is_contiguous());
            }
        }
        for endpoint in [
            numerator.lower(),
            numerator.upper(),
            divisor.lower(),
            divisor.upper(),
        ] {
            client.copy_into(
                endpoint,
                &Tensor::zeros(endpoint.shape(), dtype, client.device())?,
            )?;
        }
        for result in [&exact, &interval] {
            assert_eq!(input_bits(client, result.lower())?, expected[0]);
            assert_eq!(input_bits(client, result.upper())?, expected[1]);
        }
    }
    broadcasting::<R, C>(client, dtype)?;
    controls::<R, C>(client, dtype)
}

fn broadcasting<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let tiny = support::eta(dtype);
    let input = tensor::<R>(&[tiny, -tiny, -0.0, 3.0], &[4, 1], dtype, client.device())?;
    let divisor = tensor::<R>(&[1.0, 2.0], &[1, 2], dtype, client.device())?;
    let results = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "broadcast division identities",
        |context| {
            let numerator = context.point(&input)?;
            let divisor_interval = context.point(&divisor)?;
            Ok([
                context.div_positive(&numerator, &divisor)?,
                context.div_positive_interval(&numerator, &divisor_interval)?,
            ])
        },
    )?;
    let expected = input_bits(client, &input)?;
    for result in results {
        assert_eq!(result.lower().shape(), &[4, 2]);
        for endpoint in [result.lower(), result.upper()] {
            let bits = input_bits(client, endpoint)?;
            for row in 0..4 {
                assert_eq!(bits[row * 2], expected[row]);
            }
        }
        let lo = read(result.lower())?;
        let hi = read(result.upper())?;
        assert!(lo[1] <= 0.0 && hi[1] >= tiny && lo[1] < hi[1]);
        assert!(lo[3] <= -tiny && hi[3] >= 0.0 && lo[3] < hi[3]);
        assert_eq!(lo[7], 1.5);
        assert_eq!(hi[7], 1.5);
    }
    for shape in [&[0][..], &[0, 2][..]] {
        let empty = Tensor::<R>::empty(shape, dtype, client.device())?;
        let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
        let results = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "empty division identities",
            |context| {
                let numerator = context.point(&empty)?;
                let divisor = context.point(&one)?;
                Ok([
                    context.div_positive(&numerator, &one)?,
                    context.div_positive_interval(&numerator, &divisor)?,
                ])
            },
        )?;
        for result in results {
            assert_eq!(result.lower().shape(), shape);
            assert_eq!(result.upper().shape(), shape);
        }
    }
    Ok(())
}

fn controls<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let next_one = if dtype == DType::F32 {
        f64::from(f32::from_bits(1.0_f32.to_bits() + 1))
    } else {
        f64::from_bits(1.0_f64.to_bits() + 1)
    };
    let two = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let three = tensor::<R>(&[3.0], &[1], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let adjacent = tensor::<R>(&[next_one], &[1], dtype, client.device())?;
    let results =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "nonidentity divisors", |context| {
            let numerator = context.bounds(&two, &three)?;
            let adjacent_interval = context.point(&adjacent)?;
            let uncertain = context.bounds(&one, &adjacent)?;
            Ok([
                context.div_positive(&numerator, &adjacent)?,
                context.div_positive_interval(&numerator, &adjacent_interval)?,
                context.div_positive_interval(&numerator, &uncertain)?,
            ])
        })?;
    for result in results {
        assert!(read(result.lower())?[0] < 2.0);
        assert!(read(result.upper())?[0] >= 3.0 / next_one);
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let divisor = tensor::<R>(&[1.0, invalid], &[2], dtype, client.device())?;
        let numerator = tensor::<R>(&[support::eta(dtype), 2.0], &[2], dtype, client.device())?;
        for interval in [false, true] {
            let error = failure(with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "mixed invalid identity divisor",
                |context| {
                    let numerator = context.point(&numerator)?;
                    let result = if interval {
                        let divisor = context.point(&divisor)?;
                        context.div_positive_interval(&numerator, &divisor)?
                    } else {
                        context.div_positive(&numerator, &divisor)?
                    };
                    Ok(result)
                },
            ));
            range(error, "mixed invalid identity divisor");
        }
    }
    for (lower, upper) in [([1.0, f64::NAN], [1.0, 2.0]), ([1.0, 3.0], [1.0, 2.0])] {
        let lo = tensor::<R>(&lower, &[2], dtype, client.device())?;
        let hi = tensor::<R>(&upper, &[2], dtype, client.device())?;
        let ones = tensor::<R>(&[1.0; 2], &[2], dtype, client.device())?;
        for interval in [false, true] {
            let error = failure(with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "invalid numerator identity",
                |context| {
                    let numerator = context.bounds(&lo, &hi)?;
                    let result = if interval {
                        let divisor = context.point(&ones)?;
                        context.div_positive_interval(&numerator, &divisor)?
                    } else {
                        context.div_positive(&numerator, &ones)?
                    };
                    Ok(result)
                },
            ));
            range(error, "invalid numerator identity");
        }
    }
    Ok(())
}

#[test]
fn configured_cpu_positive_division_by_one_preserves_owned_interval_endpoints() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let base = CpuClient::new(CpuDevice::new());
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    for dtype in [DType::F32, DType::F64] {
        run::<CpuRuntime, _>(&client, dtype)?;
    }
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_positive_division_by_one_preserves_owned_interval_endpoints() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        run::<CudaRuntime, _>(&client, dtype)?;
    }
    Ok(())
}
