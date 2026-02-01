//! CUDA implementation of hypothesis testing algorithms.

use crate::stats::TensorTestResult;
use crate::stats::impl_generic::{
    pearsonr_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl, ttest_rel_impl,
};
use crate::stats::traits::HypothesisTestingAlgorithms;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        // Skip if no CUDA device available
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
}
