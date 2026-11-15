use super::fixture::{self, Client, Result, decode, input};
use numr::prelude::*;
use solvr::common::arithmetic::{CertifiedExpansion, with_interval_arithmetic};

fn components<R, C>(client: &C, value: &CertifiedExpansion<R>) -> Result<Vec<Vec<u64>>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    [value.high(), value.low(), value.abs_error()]
        .into_iter()
        .map(|tensor| Ok(fixture::bits(client, tensor)?))
        .collect()
}

fn identity<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (maximum, low) = if dtype == DType::F32 {
        (f64::from(f32::MAX), 2.0_f64.powi(-25))
    } else {
        (f64::MAX, 2.0_f64.powi(-54))
    };
    let tiny = decode(1, dtype);
    // Identity lanes include values that the general residual guard cannot certify exactly.
    let left = [
        [1.0, 0.0, 0.0],
        [-tiny, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, low, 0.125],
        [1.0, 0.0, 0.0],
        [-0.0, -0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-maximum, 0.0, 0.0],
        [1.0, -0.0, -0.0],
        [2.0, 0.0, 0.0],
        [tiny, 0.0, 0.0],
        [1.0, 0.0, tiny],
        [1.0, low, 0.0],
    ];
    let right = [
        [tiny, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, -low, tiny],
        [1.0, 0.0, 0.0],
        [maximum, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [tiny, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    for strided in [false, true] {
        for swapped in [false, true] {
            let tensors = |rows: &[[f64; 3]]| -> Result<Vec<Tensor<R>>> {
                (0..3)
                    .map(|component| {
                        Ok(input::<R>(
                            &rows.iter().map(|row| row[component]).collect::<Vec<_>>(),
                            dtype,
                            client.device(),
                            strided,
                        )?)
                    })
                    .collect()
            };
            let raw_left = tensors(&left)?;
            let raw_right = tensors(&right)?;
            if strided {
                assert!(!raw_left[0].is_contiguous());
            }
            let (a, b, product) = with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "exact-one expansion product",
                |context| {
                    assert_eq!(context.dtype(), dtype);
                    let minimum_normal = if dtype == DType::F32 {
                        f64::from(f32::MIN_POSITIVE)
                    } else {
                        f64::MIN_POSITIVE
                    };
                    assert_eq!(context.minimum_normal()?, minimum_normal);
                    let a = context.expansion_bounds(&raw_left[0], &raw_left[1], &raw_left[2])?;
                    let b =
                        context.expansion_bounds(&raw_right[0], &raw_right[1], &raw_right[2])?;
                    let product = if swapped {
                        context.expansion_mul(&b, &a)?
                    } else {
                        context.expansion_mul(&a, &b)?
                    };
                    Ok((a, b, product))
                },
            )?;
            let before = components(client, &product)?;
            let a_components = components(client, &a)?;
            let b_components = components(client, &b)?;
            for lane in 0..9 {
                let from_right = [0, 2, 4, 6].contains(&lane) || (lane == 8 && !swapped);
                let expected = if from_right {
                    &b_components
                } else {
                    &a_components
                };
                for component in 0..3 {
                    assert_eq!(
                        before[component][lane], expected[component][lane],
                        "dtype={dtype:?}, swapped={swapped}, lane={lane}, component={component}"
                    );
                }
            }
            assert_eq!(decode(before[0][9], dtype), 6.0);
            assert_eq!(decode(before[2][9], dtype), 0.0);
            assert!(decode(before[2][10], dtype) > 0.0);
            assert!(decode(before[2][11], dtype) > 0.0);
            assert_eq!(decode(before[1][12], dtype), 2.0 * low);
            for tensor in [product.high(), product.low(), product.abs_error()] {
                assert_eq!(tensor.dtype(), dtype);
                assert_eq!(tensor.shape(), &[left.len()]);
                assert!(tensor.device().is_same(client.device()));
                assert!(tensor.is_contiguous());
            }
            for value in [&a, &b] {
                for tensor in [value.high(), value.low(), value.abs_error()] {
                    client.copy_into(
                        tensor,
                        &Tensor::zeros(tensor.shape(), dtype, client.device())?,
                    )?;
                }
            }
            assert_eq!(components(client, &product)?, before);
        }
        let empty = input::<R>(&[], dtype, client.device(), strided)?;
        let result = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "empty identity product",
            |context| {
                let value = context.expansion_point(&empty)?;
                context.expansion_mul(&value, &value)
            },
        )?;
        for tensor in [result.high(), result.low(), result.abs_error()] {
            assert_eq!(tensor.shape(), &[0]);
            assert_eq!(tensor.dtype(), dtype);
        }
    }
    Ok(())
}

#[test]
fn configured_cpu_exact_one_products_preserve_complete_owned_expansions() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let base = CpuRuntime::default_client(&CpuDevice::new());
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    for dtype in [DType::F32, DType::F64] {
        identity::<CpuRuntime, _>(&client, dtype)?;
    }
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_exact_one_products_preserve_complete_owned_expansions() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        identity::<CudaRuntime, _>(&client, dtype)?;
    }
    Ok(())
}
