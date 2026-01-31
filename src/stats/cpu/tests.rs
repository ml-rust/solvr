//! Tests for CPU statistics implementation.

use crate::stats::StatisticsAlgorithms;
use crate::stats::helpers::extract_scalar;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn test_describe() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let stats = client.describe(&data).unwrap();

    assert_eq!(stats.nobs, 5);

    let min_val = extract_scalar(&stats.min).unwrap();
    let max_val = extract_scalar(&stats.max).unwrap();
    let mean_val = extract_scalar(&stats.mean).unwrap();

    assert!((min_val - 1.0).abs() < 1e-10);
    assert!((max_val - 5.0).abs() < 1e-10);
    assert!((mean_val - 3.0).abs() < 1e-10);
}

#[test]
fn test_percentile() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let p50 = StatisticsAlgorithms::percentile(&client, &data, 50.0).unwrap();
    let p50_val = extract_scalar(&p50).unwrap();
    assert!((p50_val - 3.0).abs() < 1e-10);
}

#[test]
fn test_zscore() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let z = client.zscore(&data).unwrap();
    let z_data: Vec<f64> = z.to_vec();

    // Mean of z-scores should be ~0
    let z_mean: f64 = z_data.iter().sum::<f64>() / z_data.len() as f64;
    assert!(z_mean.abs() < 1e-10);
}

#[test]
fn test_ttest_1samp() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.2f64, 1.5, 1.3, 1.4, 1.6], &[5], &device);

    let result = client.ttest_1samp(&data, 1.0).unwrap();
    let stat = extract_scalar(&result.statistic).unwrap();
    let pval = extract_scalar(&result.pvalue).unwrap();

    // Should reject null (mean != 1.0)
    assert!(stat > 0.0);
    assert!(pval < 0.05);
}

#[test]
fn test_ttest_ind() {
    let (client, device) = setup();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0, 4.0, 5.0, 6.0], &[5], &device);

    let result = client.ttest_ind(&a, &b).unwrap();
    let stat = extract_scalar(&result.statistic).unwrap();
    let pval = extract_scalar(&result.pvalue).unwrap();

    // Groups differ by 1, should have negative t-stat (a < b)
    assert!(stat < 0.0);
    // With small samples and 1 unit difference, may or may not be significant
    assert!(pval > 0.0 && pval < 1.0);
}

#[test]
fn test_pearsonr() {
    let (client, device) = setup();
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

    let result = client.pearsonr(&x, &y).unwrap();
    let r = extract_scalar(&result.statistic).unwrap();
    let pval = extract_scalar(&result.pvalue).unwrap();

    // Perfect positive correlation
    assert!((r - 1.0).abs() < 1e-10);
    assert!(pval < 0.01);
}

#[test]
fn test_linregress() {
    let (client, device) = setup();
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

    let result = client.linregress(&x, &y).unwrap();

    // Perfect linear relationship: y = 2x
    assert!((result.slope - 2.0).abs() < 1e-10);
    assert!((result.intercept - 0.0).abs() < 1e-10);
    assert!((result.rvalue - 1.0).abs() < 1e-10);
}

#[test]
fn test_skewness() {
    let (client, device) = setup();
    // Symmetric data should have ~0 skewness
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let skew = client.skewness(&data).unwrap();
    let skew_val = extract_scalar(&skew).unwrap();
    assert!(skew_val.abs() < 0.1);
}

#[test]
fn test_kurtosis() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[10],
        &device,
    );

    let kurt = client.kurtosis(&data).unwrap();
    let kurt_val = extract_scalar(&kurt).unwrap();
    // Uniform-ish data has negative excess kurtosis
    assert!(kurt_val < 0.0);
}

#[test]
fn test_iqr() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let iqr = client.iqr(&data).unwrap();
    let iqr_val = extract_scalar(&iqr).unwrap();
    // Q3 - Q1 for [1,2,3,4,5] is 4-2 = 2
    assert!((iqr_val - 2.0).abs() < 1e-10);
}

#[test]
fn test_sem() {
    let (client, device) = setup();
    let data = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let sem = client.sem(&data).unwrap();
    let sem_val = extract_scalar(&sem).unwrap();
    // std / sqrt(n) for this data
    assert!(sem_val > 0.0);
}
