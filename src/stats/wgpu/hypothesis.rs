//! WebGPU implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    bartlett_impl, f_oneway_impl, friedmanchisquare_impl, kruskal_impl, levene_impl,
    normaltest_impl, pearsonr_impl, shapiro_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl,
    ttest_rel_impl,
};
use crate::stats::traits::{HypothesisTestingAlgorithms, LeveneCenter};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl HypothesisTestingAlgorithms<WgpuRuntime> for WgpuClient {
    fn ttest_1samp(
        &self,
        x: &Tensor<WgpuRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        spearmanr_impl(self, x, y)
    }

    fn f_oneway(&self, groups: &[&Tensor<WgpuRuntime>]) -> Result<TensorTestResult<WgpuRuntime>> {
        f_oneway_impl(self, groups)
    }

    fn kruskal(&self, groups: &[&Tensor<WgpuRuntime>]) -> Result<TensorTestResult<WgpuRuntime>> {
        kruskal_impl(self, groups)
    }

    fn friedmanchisquare(
        &self,
        groups: &[&Tensor<WgpuRuntime>],
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        friedmanchisquare_impl(self, groups)
    }

    fn shapiro(&self, x: &Tensor<WgpuRuntime>) -> Result<TensorTestResult<WgpuRuntime>> {
        shapiro_impl(self, x)
    }

    fn normaltest(&self, x: &Tensor<WgpuRuntime>) -> Result<TensorTestResult<WgpuRuntime>> {
        normaltest_impl(self, x)
    }

    fn levene(
        &self,
        groups: &[&Tensor<WgpuRuntime>],
        center: LeveneCenter,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        levene_impl(self, groups, center)
    }

    fn bartlett(&self, groups: &[&Tensor<WgpuRuntime>]) -> Result<TensorTestResult<WgpuRuntime>> {
        bartlett_impl(self, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_ttest_1samp_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let data = Tensor::<WgpuRuntime>::from_slice(&[1.2f64, 1.5, 1.3, 1.4, 1.6], &[5], &device);
        let result = client.ttest_1samp(&data, 1.0).unwrap();

        let stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(stat > 0.0);
        assert!(pval < 0.05);
    }
}
