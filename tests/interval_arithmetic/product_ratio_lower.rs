use super::support::{self, Client, Result, input_bits, read, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

fn lower(value: f64, truth: f64, maximum: f64) {
    assert!(
        value.is_finite() && value > 0.0 && value <= truth,
        "lower={value}, truth={truth}"
    );
    if value != maximum {
        let fraction = value.to_bits() & ((1_u64 << 52) - 1);
        let exponent = (value.to_bits() >> 52) & 0x7ff;
        assert!(if exponent == 0 {
            fraction.count_ones() == 1
        } else {
            fraction == 0
        });
    }
}

fn component<R, C>(
    client: &C,
    values: &[f64],
    dtype: DType,
    strided: bool,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    if strided {
        let storage: Vec<_> = values.iter().flat_map(|&value| [73.0, value]).collect();
        let view = tensor::<R>(&storage, &[values.len(), 2], dtype, client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(view.offset() > 0 && !view.is_contiguous());
        Ok(view)
    } else {
        tensor::<R>(values, &[values.len()], dtype, client.device())
    }
}

fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let exponent = if dtype == DType::F32 { 100 } else { 900 };
    let largest_power = if dtype == DType::F32 { 127 } else { 1023 };
    let maximum = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let p = 2.0_f64.powi(exponent);
    let s = 2.0_f64.powi(-exponent);
    let top = 2.0_f64.powi(largest_power);
    let eta = support::eta(dtype);
    let values = [
        [p, p, s, s, eta, maximum, eta, top, p],
        [p, 1.0, s, 1.0, 1.0, maximum, 1.0, top, p],
        [p, s, s, p, eta, maximum, 1.0, 1.0, p],
        [1.0, p, 1.0, s, 1.0, 1.0, 1.0, 1.0, p],
    ];
    let truths = [p, p, s, s, 1.0, maximum, eta, maximum, 1.0];
    for strided in [false, true] {
        for reversed in [false, true] {
            let factors = values
                .iter()
                .map(|v| component::<R, C>(client, v, dtype, strided))
                .collect::<numr::error::Result<Vec<_>>>()?;
            let denominators = if reversed {
                [&factors[3], &factors[2]]
            } else {
                [&factors[2], &factors[3]]
            };
            let result = with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "positive product ratio range recovery",
                |context| {
                    context.positive_product_ratio_lower(&[&factors[0], &factors[1]], &denominators)
                },
            )?;
            assert_eq!(result.shape(), &[9]);
            assert_eq!(result.dtype(), dtype);
            assert!(result.device().is_same(client.device()));
            assert!(result.is_contiguous());
            let observed = read(&result)?;
            for (&actual, truth) in observed.iter().zip(truths) {
                lower(actual, truth, maximum);
            }
            assert_eq!(observed[6], eta);
            assert_eq!(observed[7], maximum);
            let expected = input_bits(client, &result)?;
            for factor in &factors {
                client.copy_into(
                    factor,
                    &Tensor::zeros(factor.shape(), dtype, client.device())?,
                )?;
            }
            assert_eq!(input_bits(client, &result)?, expected);
            client.copy_into(
                &result,
                &Tensor::zeros(result.shape(), dtype, client.device())?,
            )?;
            for factor in &factors {
                assert!(input_bits(client, factor)?.iter().all(|&bits| bits == 0));
            }
            let fresh = values
                .iter()
                .map(|v| component::<R, C>(client, v, dtype, strided))
                .collect::<numr::error::Result<Vec<_>>>()?;
            let fresh_denominators = if reversed {
                [&fresh[3], &fresh[2]]
            } else {
                [&fresh[2], &fresh[3]]
            };
            let independent = with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "independent positive product ratio",
                |context| {
                    context
                        .positive_product_ratio_lower(&[&fresh[0], &fresh[1]], &fresh_denominators)
                },
            )?;
            assert_eq!(input_bits(client, &independent)?, expected);
        }
    }
    let minimum = tensor::<R>(&[eta], &[], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    for denominator in [&[][..], &[&one][..]] {
        let result = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "minimum positive dyadic lower",
            |context| context.positive_product_ratio_lower(&[&minimum], denominator),
        )?;
        assert_eq!(input_bits(client, &result)?, vec![1]);
    }
    let neutral = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "empty factor product identity",
        |context| context.positive_product_ratio_lower(&[], &[]),
    )?;
    let independent_neutral = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "owned empty factor identity",
        |context| context.positive_product_ratio_lower(&[], &[]),
    )?;
    assert_eq!(neutral.shape(), &[] as &[usize]);
    assert_eq!(neutral.dtype(), dtype);
    assert!(neutral.device().is_same(client.device()) && neutral.is_contiguous());
    assert_eq!(read(&neutral)?, vec![1.0]);
    client.copy_into(&neutral, &Tensor::zeros(&[], dtype, client.device())?)?;
    assert_eq!(read(&independent_neutral)?, vec![1.0]);
    let a = tensor::<R>(&[1.5, 1.75], &[2], dtype, client.device())?;
    let b = tensor::<R>(&[1.5, 1.0], &[2], dtype, client.device())?;
    let divisor = tensor::<R>(&[2.0, 1.5], &[2], dtype, client.device())?;
    let mantissas = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "rounded mantissa positive lower",
        |context| context.positive_product_ratio_lower(&[&a, &b], &[&divisor]),
    )?;
    let mantissas = read(&mantissas)?;
    lower(mantissas[0], 1.125, maximum);
    lower(mantissas[1], 2.0, maximum);
    assert!(mantissas[1] * 6.0 <= 7.0);
    broadcasting::<R, C>(client, dtype)?;
    super::product_ratio_lower_admission::run::<R, C>(client, dtype)
}

fn broadcasting<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let storage = tensor::<R>(&[91.0, 16.0, 91.0, 0.0625], &[2, 2], dtype, client.device())?;
    let rows = storage.narrow(1, 1, 1)?;
    assert!(rows.offset() > 0 && !rows.is_contiguous());
    let columns = tensor::<R>(&[1.0, 2.0, 4.0], &[1, 3], dtype, client.device())?;
    let denominator = tensor::<R>(&[1.0, 4.0, 16.0], &[1, 3], dtype, client.device())?;
    let scalar = tensor::<R>(&[1.0], &[], dtype, client.device())?;
    let result = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "joint ratio broadcasting",
        |context| {
            context.positive_product_ratio_lower(&[&rows, &columns, &scalar], &[&denominator])
        },
    )?;
    assert_eq!(result.shape(), &[2, 3]);
    let maximum = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    for (actual, truth) in read(&result)?
        .into_iter()
        .zip([16.0, 8.0, 4.0, 0.0625, 0.03125, 0.015625])
    {
        lower(actual, truth, maximum);
    }
    for shape in [&[0][..], &[0, 3][..]] {
        let empty = Tensor::<R>::empty(shape, dtype, client.device())?;
        let result = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "empty positive ratio",
            |context| context.positive_product_ratio_lower(&[&empty, &scalar], &[&scalar]),
        )?;
        assert_eq!(result.shape(), shape);
        assert_eq!(result.dtype(), dtype);
        assert!(result.device().is_same(client.device()));
        assert!(read(&result)?.is_empty());
    }
    let empty = Tensor::<R>::empty(&[0, 3], dtype, client.device())?;
    let result = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "empty dimension joint ratio broadcast",
        |context| {
            context.positive_product_ratio_lower(&[&empty, &columns], &[&denominator, &scalar])
        },
    )?;
    assert_eq!(result.shape(), &[0, 3]);
    assert_eq!(result.dtype(), dtype);
    assert!(result.device().is_same(client.device()) && read(&result)?.is_empty());
    Ok(())
}

#[test]
fn configured_cpu_positive_product_ratio_recovers_finite_dyadic_lowers() -> Result {
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
fn actual_cuda_positive_product_ratio_recovers_finite_dyadic_lowers() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        run::<CudaRuntime, _>(&client, dtype)?;
        let foreign = Tensor::<CudaRuntime>::empty(&[0], dtype, &CudaDevice::new(1))?;
        let one = tensor::<CudaRuntime>(&[1.0], &[], dtype, client.device())?;
        for numerator in [false, true] {
            let error = with_interval_arithmetic::<CudaRuntime, _, _, _>(
                &client,
                dtype,
                "foreign ratio factor before empty output",
                |context| {
                    if numerator {
                        context.positive_product_ratio_lower(&[&foreign], &[&one])
                    } else {
                        context.positive_product_ratio_lower(&[&one], &[&foreign])
                    }
                },
            )
            .err()
            .ok_or("Accepted a foreign-device ratio factor")?;
            assert!(matches!(
                error,
                solvr::common::arithmetic::ArithmeticError::Numr(_)
            ));
        }
    }
    Ok(())
}
