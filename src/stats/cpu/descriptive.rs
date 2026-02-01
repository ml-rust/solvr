//! CPU implementation of descriptive statistics algorithms.

use crate::stats::TensorDescriptiveStats;
use crate::stats::impl_generic::{
    describe_impl, iqr_impl, kurtosis_impl, percentile_impl, sem_impl, skewness_impl, zscore_impl,
};
use crate::stats::traits::DescriptiveStatisticsAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DescriptiveStatisticsAlgorithms<CpuRuntime> for CpuClient {
    fn describe(&self, x: &Tensor<CpuRuntime>) -> Result<TensorDescriptiveStats<CpuRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<CpuRuntime>, p: f64) -> Result<Tensor<CpuRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        sem_impl(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::distribution::{ContinuousDistribution, DiscreteDistribution};
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cpu::CpuDevice;

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

        let p50 = DescriptiveStatisticsAlgorithms::percentile(&client, &data, 50.0).unwrap();
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

    // ============================================================================
    // Distribution Batch Tensor Method Tests
    // ============================================================================

    #[test]
    fn test_normal_pdf_tensor() {
        let (client, device) = setup();
        let n = crate::stats::Normal::standard();

        let x = Tensor::<CpuRuntime>::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let pdf = n.pdf_tensor(&x, &client).unwrap();
        let pdf_data: Vec<f64> = pdf.to_vec();

        // Compare with scalar results
        assert!((pdf_data[0] - n.pdf(-2.0)).abs() < 1e-10);
        assert!((pdf_data[1] - n.pdf(-1.0)).abs() < 1e-10);
        assert!((pdf_data[2] - n.pdf(0.0)).abs() < 1e-10);
        assert!((pdf_data[3] - n.pdf(1.0)).abs() < 1e-10);
        assert!((pdf_data[4] - n.pdf(2.0)).abs() < 1e-10);

        // PDF at 0 for standard normal
        assert!((pdf_data[2] - 0.3989422804014327).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_tensor() {
        let (client, device) = setup();
        let n = crate::stats::Normal::standard();

        let x = Tensor::<CpuRuntime>::from_slice(&[-2.0, 0.0, 2.0], &[3], &device);
        let cdf = n.cdf_tensor(&x, &client).unwrap();
        let cdf_data: Vec<f64> = cdf.to_vec();

        // CDF at 0 should be 0.5
        assert!((cdf_data[1] - 0.5).abs() < 1e-10);

        // Symmetry: CDF(-2) + CDF(2) should be ~1
        assert!((cdf_data[0] + cdf_data[2] - 1.0).abs() < 1e-10);

        // Compare with scalar
        assert!((cdf_data[0] - n.cdf(-2.0)).abs() < 1e-10);
        assert!((cdf_data[2] - n.cdf(2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_normal_ppf_tensor() {
        let (client, device) = setup();
        let n = crate::stats::Normal::standard();

        let p = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.5, 0.9], &[3], &device);
        let ppf = n.ppf_tensor(&p, &client).unwrap();
        let ppf_data: Vec<f64> = ppf.to_vec();

        // PPF(0.5) = 0 for standard normal
        assert!(ppf_data[1].abs() < 1e-10);

        // Symmetry: PPF(0.1) = -PPF(0.9)
        assert!((ppf_data[0] + ppf_data[2]).abs() < 1e-10);

        // Compare with scalar
        assert!((ppf_data[0] - n.ppf(0.1).unwrap()).abs() < 1e-10);
        assert!((ppf_data[2] - n.ppf(0.9).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_normal_sf_tensor() {
        let (client, device) = setup();
        let n = crate::stats::Normal::standard();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.96], &[2], &device);
        let sf = n.sf_tensor(&x, &client).unwrap();
        let sf_data: Vec<f64> = sf.to_vec();

        // SF(0) = 0.5
        assert!((sf_data[0] - 0.5).abs() < 1e-10);

        // SF(1.96) ~ 0.025
        assert!((sf_data[1] - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_uniform_tensor() {
        let (client, device) = setup();
        let u = crate::stats::Uniform::new(0.0, 1.0).unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], &device);

        // PDF should be constant = 1 over [0, 1]
        let pdf = u.pdf_tensor(&x, &client).unwrap();
        let pdf_data: Vec<f64> = pdf.to_vec();
        for &p in &pdf_data {
            assert!((p - 1.0).abs() < 1e-10);
        }

        // CDF should equal x for x in [0, 1]
        let cdf = u.cdf_tensor(&x, &client).unwrap();
        let cdf_data: Vec<f64> = cdf.to_vec();
        let x_data: Vec<f64> = x.to_vec();
        for (c, x) in cdf_data.iter().zip(x_data.iter()) {
            assert!((c - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_binomial_pmf_tensor() {
        let (client, device) = setup();
        let b = crate::stats::Binomial::new(10, 0.5).unwrap();

        let k = Tensor::<CpuRuntime>::from_slice(&[0.0, 5.0, 10.0], &[3], &device);
        let pmf = b.pmf_tensor(&k, &client).unwrap();
        let pmf_data: Vec<f64> = pmf.to_vec();

        // Compare with scalar
        assert!((pmf_data[0] - b.pmf(0)).abs() < 1e-10);
        assert!((pmf_data[1] - b.pmf(5)).abs() < 1e-10);
        assert!((pmf_data[2] - b.pmf(10)).abs() < 1e-10);

        // PMF(5) should be maximum for fair coin (most likely outcome)
        assert!(pmf_data[1] > pmf_data[0]);
        assert!(pmf_data[1] > pmf_data[2]);
    }

    #[test]
    fn test_binomial_cdf_tensor() {
        let (client, device) = setup();
        let b = crate::stats::Binomial::new(10, 0.5).unwrap();

        let k = Tensor::<CpuRuntime>::from_slice(&[0.0, 5.0, 10.0], &[3], &device);
        let cdf = b.cdf_tensor(&k, &client).unwrap();
        let cdf_data: Vec<f64> = cdf.to_vec();

        // CDF is monotonic
        assert!(cdf_data[0] < cdf_data[1]);
        assert!(cdf_data[1] < cdf_data[2]);

        // CDF(10) = 1
        assert!((cdf_data[2] - 1.0).abs() < 1e-10);

        // Compare with scalar
        assert!((cdf_data[0] - b.cdf(0)).abs() < 1e-10);
        assert!((cdf_data[1] - b.cdf(5)).abs() < 1e-6);
    }

    #[test]
    fn test_binomial_ppf_tensor() {
        let (client, device) = setup();
        let b = crate::stats::Binomial::new(10, 0.5).unwrap();

        let p = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let ppf = b.ppf_tensor(&p, &client).unwrap();
        let ppf_data: Vec<f64> = ppf.to_vec();

        // PPF(0) = 0
        assert!((ppf_data[0] - 0.0).abs() < 1e-10);

        // PPF(1) = n = 10
        assert!((ppf_data[2] - 10.0).abs() < 1e-10);

        // Compare with scalar
        assert!((ppf_data[1] - b.ppf(0.5).unwrap() as f64).abs() < 1e-10);
    }

    #[test]
    fn test_2d_tensor_batch() {
        let (client, device) = setup();
        let n = crate::stats::Normal::standard();

        // Test with 2D tensor
        let x =
            Tensor::<CpuRuntime>::from_slice(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0], &[2, 3], &device);

        let pdf = n.pdf_tensor(&x, &client).unwrap();
        let pdf_data: Vec<f64> = pdf.to_vec();

        // Shape should be preserved
        assert_eq!(pdf.shape(), &[2, 3]);

        // Values should match scalar computation
        assert!((pdf_data[0] - n.pdf(-1.0)).abs() < 1e-10);
        assert!((pdf_data[1] - n.pdf(0.0)).abs() < 1e-10);
        assert!((pdf_data[2] - n.pdf(1.0)).abs() < 1e-10);
    }
}
