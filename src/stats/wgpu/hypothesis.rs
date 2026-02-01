//! WebGPU implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    pearsonr_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl, ttest_rel_impl,
};
use crate::stats::traits::HypothesisTestingAlgorithms;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
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
