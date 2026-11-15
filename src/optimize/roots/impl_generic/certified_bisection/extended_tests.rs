use super::correction_fixtures::{self, CorrectionCase};
use super::extended::{CorrectionInputs, intersect};
use crate::DType;
use crate::common::arithmetic::{ArithmeticError, ArithmeticResult, with_interval_arithmetic};
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn represented(dtype: DType, bits: u64) -> f64 {
    match dtype {
        DType::F32 => f64::from(f32::from_bits(bits as u32)),
        _ => f64::from_bits(bits),
    }
}

fn tensor<R: Runtime<DType = DType>>(
    device: &R::Device,
    dtype: DType,
    bits: &[u64],
) -> ArithmeticResult<Tensor<R>> {
    Ok(match dtype {
        DType::F32 => Tensor::from_slice(
            &bits
                .iter()
                .map(|&value| f32::from_bits(value as u32))
                .collect::<Vec<_>>(),
            &[bits.len()],
            device,
        )?,
        DType::F64 => Tensor::from_slice(
            &bits
                .iter()
                .map(|&value| f64::from_bits(value))
                .collect::<Vec<_>>(),
            &[bits.len()],
            device,
        )?,
        _ => return Err(ArithmeticError::UnsupportedDType { dtype }),
    })
}

fn check_cases<R, C>(client: &C, dtype: DType, cases: &[CorrectionCase]) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let column = |index| {
        tensor::<R>(
            client.device(),
            dtype,
            &cases
                .iter()
                .map(|case| case.operands[index])
                .collect::<Vec<_>>(),
        )
    };
    let lower = column(0)?;
    let upper = column(1)?;
    let point = column(2)?;
    let residual_lower = column(3)?;
    let residual_upper = column(4)?;
    let derivative_lower = column(5)?;
    let active_bits = cases
        .iter()
        .map(|case| {
            if !case.active {
                0
            } else if dtype == DType::F32 {
                0x3f800000
            } else {
                0x3ff0000000000000
            }
        })
        .collect::<Vec<_>>();
    let active = tensor::<R>(client.device(), dtype, &active_bits)?;
    let result = with_interval_arithmetic(client, dtype, "certified correction fixtures", |ctx| {
        intersect(
            ctx,
            CorrectionInputs {
                lower: &lower,
                upper: &upper,
                point: &point,
                residual_lower: &residual_lower,
                residual_upper: &residual_upper,
                derivative_lower: &derivative_lower,
                active: &active,
            },
        )
    })?;
    let values = |value: &Tensor<R>| -> ArithmeticResult<Vec<f64>> {
        Ok(match dtype {
            DType::F32 => value
                .try_to_vec::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect(),
            _ => value.try_to_vec::<f64>()?,
        })
    };
    let result_lower = values(result.lower())?;
    let result_upper = values(result.upper())?;
    for (index, case) in cases.iter().enumerate() {
        let lo = result_lower[index];
        let hi = result_upper[index];
        assert!(lo.is_finite() && hi.is_finite() && lo <= hi, "{}", case.id);
        assert!(lo >= represented(dtype, case.operands[0]), "{}", case.id);
        assert!(hi <= represented(dtype, case.operands[1]), "{}", case.id);
        assert!(lo <= represented(dtype, case.enclosure[0]), "{}", case.id);
        assert!(hi >= represented(dtype, case.enclosure[1]), "{}", case.id);
        for (actual, exact) in [(lo, case.exact[0]), (hi, case.exact[1])] {
            if let Some(bits) = exact {
                assert_eq!(
                    actual.to_bits(),
                    represented(dtype, bits).to_bits(),
                    "{}",
                    case.id
                );
            }
        }
    }
    check_malformed::<R, C>(client, dtype)?;
    Ok(())
}

fn check_malformed<R, C>(client: &C, dtype: DType) -> ArithmeticResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let constant = |value| Tensor::<R>::full_scalar(&[1], dtype, value, client.device());
    let lower = constant(-1.0)?;
    let upper = constant(1.0)?;
    // Columns contain point, residual lower, residual upper, derivative floor, and active mask.
    for row in [
        [f64::NAN, -0.25, 0.25, 1.0, 1.0],
        [2.0, -0.25, 0.25, 1.0, 1.0],
        [0.0, f64::NAN, 0.25, 1.0, 1.0],
        [0.0, -0.25, f64::INFINITY, 1.0, 1.0],
        [0.0, 0.25, -0.25, 1.0, 1.0],
        [0.0, -0.25, 0.25, 0.0, 1.0],
        [0.0, -0.25, 0.25, -1.0, 1.0],
        [0.0, -0.25, 0.25, f64::NAN, 1.0],
        [0.0, -0.25, 0.25, 1.0, f64::NAN],
        [0.0, -0.25, 0.25, 1.0, 0.5],
    ] {
        let point = constant(row[0])?;
        let a = constant(row[1])?;
        let b = constant(row[2])?;
        let kappa = constant(row[3])?;
        let active = constant(row[4])?;
        let result = with_interval_arithmetic(client, dtype, "malformed correction", |ctx| {
            intersect(
                ctx,
                CorrectionInputs {
                    lower: &lower,
                    upper: &upper,
                    point: &point,
                    residual_lower: &a,
                    residual_upper: &b,
                    derivative_lower: &kappa,
                    active: &active,
                },
            )
        });
        assert!(matches!(result, Err(ArithmeticError::InvalidBounds)));
    }
    Ok(())
}

#[test]
fn cpu_extended_corrections_enclose_frozen_truth() -> ArithmeticResult<()> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    check_cases::<CpuRuntime, _>(&client, DType::F32, correction_fixtures::F32)?;
    check_cases::<CpuRuntime, _>(&client, DType::F64, correction_fixtures::F64)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_extended_corrections_enclose_frozen_truth() -> ArithmeticResult<()> {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    check_cases::<CudaRuntime, _>(&client, DType::F32, correction_fixtures::F32)?;
    check_cases::<CudaRuntime, _>(&client, DType::F64, correction_fixtures::F64)
}
