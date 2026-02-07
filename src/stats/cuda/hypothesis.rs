//! CUDA implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    bartlett_impl, f_oneway_impl, friedmanchisquare_impl, kruskal_impl, levene_impl,
    normaltest_impl, pearsonr_impl, shapiro_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl,
    ttest_rel_impl,
};
use crate::stats::traits::{HypothesisTestingAlgorithms, LeveneCenter};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl HypothesisTestingAlgorithms<CudaRuntime> for CudaClient {
    fn ttest_1samp(
        &self,
        x: &Tensor<CudaRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        spearmanr_impl(self, x, y)
    }

    fn f_oneway(&self, groups: &[&Tensor<CudaRuntime>]) -> Result<TensorTestResult<CudaRuntime>> {
        f_oneway_impl(self, groups)
    }

    fn kruskal(&self, groups: &[&Tensor<CudaRuntime>]) -> Result<TensorTestResult<CudaRuntime>> {
        kruskal_impl(self, groups)
    }

    fn friedmanchisquare(
        &self,
        groups: &[&Tensor<CudaRuntime>],
    ) -> Result<TensorTestResult<CudaRuntime>> {
        friedmanchisquare_impl(self, groups)
    }

    fn shapiro(&self, x: &Tensor<CudaRuntime>) -> Result<TensorTestResult<CudaRuntime>> {
        shapiro_impl(self, x)
    }

    fn normaltest(&self, x: &Tensor<CudaRuntime>) -> Result<TensorTestResult<CudaRuntime>> {
        normaltest_impl(self, x)
    }

    fn levene(
        &self,
        groups: &[&Tensor<CudaRuntime>],
        center: LeveneCenter,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        levene_impl(self, groups, center)
    }

    fn bartlett(&self, groups: &[&Tensor<CudaRuntime>]) -> Result<TensorTestResult<CudaRuntime>> {
        bartlett_impl(self, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_ttest_1samp_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let data = Tensor::<CudaRuntime>::from_slice(&[1.2f64, 1.5, 1.3, 1.4, 1.6], &[5], &device);
        let result = client.ttest_1samp(&data, 1.0).unwrap();

        let stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(stat > 0.0);
        assert!(pval < 0.05);
    }

    #[test]
    fn test_f_oneway_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let b =
            Tensor::<CudaRuntime>::from_slice(&[10.0f64, 11.0, 12.0, 13.0, 14.0], &[5], &device);

        let result = client.f_oneway(&[&a, &b]).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();
        assert!(pval < 0.01);
    }
}
