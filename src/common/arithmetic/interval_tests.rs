use super::*;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

#[test]
fn exact_zero_identities_retain_signed_zero_and_finite_extremes() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let zero_tensor = Tensor::from_slice(&[-0.0f64], &[], &device)?;
    let input = Tensor::from_slice(&[f64::MAX, -f64::MAX, 1.0, 0.0], &[2, 2], &device)?;
    let zero = arithmetic.point(&zero_tensor)?;
    let values = arithmetic.point(&input)?;
    let sum = arithmetic.add(&zero, &values)?;
    let difference = arithmetic.sub(&values, &zero)?;
    let product = arithmetic.mul(&zero, &values)?;
    let denominator = Tensor::full_scalar(&[], DType::F64, 3.0, &device)?;
    let quotient = arithmetic.div_positive(&zero, &denominator)?;
    arithmetic.check("zero identities")?;
    for output in [&sum.lower, &sum.upper, &difference.lower, &difference.upper] {
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.try_to_vec::<f64>()?, input.try_to_vec::<f64>()?);
    }
    for output in [
        &product.lower,
        &product.upper,
        &quotient.lower,
        &quotient.upper,
    ] {
        assert!(
            output
                .try_to_vec::<f64>()?
                .iter()
                .all(|value| value.to_bits() == (-0.0f64).to_bits())
        );
    }
    Ok(())
}

#[test]
fn signed_products_enclose_all_four_corners() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let constant = |value| Tensor::full_scalar(&[], DType::F64, value, &device);
    let a = arithmetic.bounds(constant(-2.0)?, constant(3.0)?)?;
    let b = arithmetic.bounds(constant(-5.0)?, constant(7.0)?)?;
    let product = arithmetic.mul(&a, &b)?;
    arithmetic.check("signed products")?;
    assert!(product.lower.item::<f64>()? <= -15.0);
    assert!(product.upper.item::<f64>()? >= 21.0);
    Ok(())
}

#[test]
fn nonzero_underflow_retains_an_outward_subnormal_bound() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let tiny = Tensor::from_slice(&[f64::from_bits(1)], &[], &device)?;
    let half = Tensor::full_scalar(&[], DType::F64, 0.5, &device)?;
    let tiny = arithmetic.point(&tiny)?;
    let half = arithmetic.point(&half)?;
    let product = arithmetic.mul(&tiny, &half)?;
    arithmetic.check("subnormal products")?;
    assert_eq!(product.upper.item::<f64>()?.to_bits(), 1);
    assert!(product.lower.item::<f64>()? <= 0.0);
    Ok(())
}

#[test]
fn invalid_positive_divisor_rejects_at_the_stage_check() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let zero = Tensor::zeros(&[], DType::F64, &device)?;
    let numerator = arithmetic.point(&zero)?;
    let quotient = arithmetic.div_positive(&numerator, &zero)?;
    assert_eq!(quotient.lower.item::<f64>()?, 0.0);
    assert!(matches!(
        arithmetic.check("positive division"),
        Err(ArithmeticError::Range {
            stage: "positive division"
        })
    ));
    Ok(())
}

#[test]
fn positive_interval_division_encloses_signed_quotient_corners() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let a = arithmetic.bounds(
        Tensor::from_slice(&[-4.0f64, -1.0, 2.0], &[3], &device)?,
        Tensor::from_slice(&[-2.0f64, 3.0, 5.0], &[3], &device)?,
    )?;
    let b = arithmetic.bounds(
        Tensor::full_scalar(&[3], DType::F64, 0.5, &device)?,
        Tensor::full_scalar(&[3], DType::F64, 2.0, &device)?,
    )?;
    let result = arithmetic.div_positive_interval(&a, &b)?;
    arithmetic.check("positive interval quotient")?;
    let lower = result.lower.try_to_vec::<f64>()?;
    let upper = result.upper.try_to_vec::<f64>()?;
    for (index, (lo, hi)) in [(-8.0, -1.0), (-2.0, 6.0), (1.0, 10.0)]
        .into_iter()
        .enumerate()
    {
        assert!(lower[index] <= lo);
        assert!(upper[index] >= hi);
    }
    Ok(())
}

#[test]
fn positive_interval_division_preserves_underflow_and_rejects_zero_denominators()
-> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let tiny = Tensor::from_slice(&[f64::from_bits(1)], &[], &device)?;
    let two = Tensor::full_scalar(&[], DType::F64, 2.0, &device)?;
    let tiny = arithmetic.point(&tiny)?;
    let two = arithmetic.point(&two)?;
    let quotient = arithmetic.div_positive_interval(&tiny, &two)?;
    arithmetic.check("subnormal interval quotient")?;
    assert_eq!(quotient.upper.item::<f64>()?.to_bits(), 1);
    assert!(quotient.lower.item::<f64>()? <= 0.0);

    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let negative_zero = Tensor::from_slice(&[-0.0f64], &[], &device)?;
    let zero = arithmetic.point(&negative_zero)?;
    let invalid = arithmetic.bounds(
        Tensor::zeros(&[], DType::F64, &device)?,
        Tensor::ones(&[], DType::F64, &device)?,
    )?;
    let quotient = arithmetic.div_positive_interval(&zero, &invalid)?;
    assert_eq!(quotient.lower.item::<f64>()?.to_bits(), (-0.0f64).to_bits());
    assert_eq!(quotient.upper.item::<f64>()?.to_bits(), (-0.0f64).to_bits());
    assert!(matches!(
        arithmetic.check("zero interval denominator"),
        Err(ArithmeticError::Range { .. })
    ));
    Ok(())
}

#[test]
fn square_root_intervals_enclose_exact_and_irrational_roots() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let input = Tensor::from_slice(&[-0.0f64, 2.0, 9.0, f64::from_bits(1)], &[4], &device)?;
    let input = arithmetic.point(&input)?;
    let root = arithmetic.sqrt_nonnegative(&input)?;
    arithmetic.check("nonnegative square roots")?;
    let lower = root.lower.try_to_vec::<f64>()?;
    let upper = root.upper.try_to_vec::<f64>()?;
    assert_eq!(lower[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(upper[0].to_bits(), (-0.0f64).to_bits());
    assert!(lower[1] <= f64::from_bits(0x3ff6a09e667f3bcc));
    assert!(upper[1] >= f64::from_bits(0x3ff6a09e667f3bcd));
    assert!(lower[2] <= 3.0 && upper[2] >= 3.0);
    let exact_subnormal_root = f64::from_bits(0x1e60000000000000);
    assert!(lower[3] <= exact_subnormal_root && upper[3] >= exact_subnormal_root);
    Ok(())
}

#[test]
fn negative_square_root_intervals_reject_without_nonfinite_candidates() -> ArithmeticResult<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
    let input = arithmetic.bounds(
        Tensor::full_scalar(&[], DType::F64, -1.0, &device)?,
        Tensor::ones(&[], DType::F64, &device)?,
    )?;
    let root = arithmetic.sqrt_nonnegative(&input)?;
    assert_eq!(root.lower.item::<f64>()?, 0.0);
    assert_eq!(root.upper.item::<f64>()?, 0.0);
    assert!(matches!(
        arithmetic.check("negative root domain"),
        Err(ArithmeticError::Range { .. })
    ));
    Ok(())
}
