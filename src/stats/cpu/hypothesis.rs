//! CPU implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    pearsonr_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl, ttest_rel_impl,
};
use crate::stats::traits::HypothesisTestingAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl HypothesisTestingAlgorithms<CpuRuntime> for CpuClient {
    fn ttest_1samp(
        &self,
        x: &Tensor<CpuRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        spearmanr_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
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
}
