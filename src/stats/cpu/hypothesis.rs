//! CPU implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    bartlett_impl, f_oneway_impl, friedmanchisquare_impl, kruskal_impl, levene_impl,
    normaltest_impl, pearsonr_impl, shapiro_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl,
    ttest_rel_impl,
};
use crate::stats::traits::{HypothesisTestingAlgorithms, LeveneCenter};
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

    fn f_oneway(&self, groups: &[&Tensor<CpuRuntime>]) -> Result<TensorTestResult<CpuRuntime>> {
        f_oneway_impl(self, groups)
    }

    fn kruskal(&self, groups: &[&Tensor<CpuRuntime>]) -> Result<TensorTestResult<CpuRuntime>> {
        kruskal_impl(self, groups)
    }

    fn friedmanchisquare(
        &self,
        groups: &[&Tensor<CpuRuntime>],
    ) -> Result<TensorTestResult<CpuRuntime>> {
        friedmanchisquare_impl(self, groups)
    }

    fn shapiro(&self, x: &Tensor<CpuRuntime>) -> Result<TensorTestResult<CpuRuntime>> {
        shapiro_impl(self, x)
    }

    fn normaltest(&self, x: &Tensor<CpuRuntime>) -> Result<TensorTestResult<CpuRuntime>> {
        normaltest_impl(self, x)
    }

    fn levene(
        &self,
        groups: &[&Tensor<CpuRuntime>],
        center: LeveneCenter,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        levene_impl(self, groups, center)
    }

    fn bartlett(&self, groups: &[&Tensor<CpuRuntime>]) -> Result<TensorTestResult<CpuRuntime>> {
        bartlett_impl(self, groups)
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

        assert!(stat < 0.0);
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

        assert!((r - 1.0).abs() < 1e-10);
        assert!(pval < 0.01);
    }

    #[test]
    fn test_f_oneway() {
        let (client, device) = setup();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0, 4.0, 5.0, 6.0], &[5], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[10.0f64, 11.0, 12.0, 13.0, 14.0], &[5], &device);

        let result = client.f_oneway(&[&a, &b, &c]).unwrap();
        let f_stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        // Group c is very different, should have high F and low p
        assert!(f_stat > 1.0);
        assert!(pval < 0.01);
    }

    #[test]
    fn test_kruskal() {
        let (client, device) = setup();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10.0f64, 11.0, 12.0, 13.0, 14.0], &[5], &device);

        let result = client.kruskal(&[&a, &b]).unwrap();
        let h = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(h > 0.0);
        assert!(pval < 0.05);
    }

    #[test]
    fn test_friedmanchisquare() {
        let (client, device) = setup();
        // 5 subjects, 3 treatments
        let t1 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 3.0, 4.0, 5.0, 6.0], &[5], &device);
        let t3 =
            Tensor::<CpuRuntime>::from_slice(&[10.0f64, 11.0, 12.0, 13.0, 14.0], &[5], &device);

        let result = client.friedmanchisquare(&[&t1, &t2, &t3]).unwrap();
        let chi2 = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(chi2 > 0.0);
        assert!(pval < 0.05);
    }

    #[test]
    fn test_shapiro() {
        let (client, device) = setup();
        // Data from a normal distribution
        let data = Tensor::<CpuRuntime>::from_slice(
            &[-1.2f64, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0, -0.8, 0.2],
            &[10],
            &device,
        );

        let result = client.shapiro(&data).unwrap();
        let w = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        // W should be close to 1 for normal data
        assert!(w > 0.8);
        // p-value should not reject normality
        assert!(pval > 0.01);
    }

    #[test]
    fn test_normaltest() {
        let (client, device) = setup();
        // Roughly normal data (n=30)
        let data: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) / 5.0).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[30], &device);

        let result = client.normaltest(&x).unwrap();
        let k2 = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(k2 >= 0.0);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_levene() {
        let (client, device) = setup();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 10.0, 1.0, 10.0, 1.0], &[5], &device);

        // b has much higher variance
        let result = client.levene(&[&a, &b], LeveneCenter::Median).unwrap();
        let f_stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(f_stat > 0.0);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_bartlett() {
        let (client, device) = setup();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 10.0, 1.0, 10.0, 1.0, 10.0], &[6], &device);

        let result = client.bartlett(&[&a, &b]).unwrap();
        let t_stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(t_stat > 0.0);
        assert!(pval >= 0.0 && pval <= 1.0);
    }
}
