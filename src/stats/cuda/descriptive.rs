//! CUDA implementation of descriptive statistics algorithms.

use crate::stats::TensorDescriptiveStats;
use crate::stats::impl_generic::{
    describe_impl, iqr_impl, kurtosis_impl, percentile_impl, sem_impl, skewness_impl, zscore_impl,
};
use crate::stats::traits::DescriptiveStatisticsAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl DescriptiveStatisticsAlgorithms<CudaRuntime> for CudaClient {
    fn describe(&self, x: &Tensor<CudaRuntime>) -> Result<TensorDescriptiveStats<CudaRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<CudaRuntime>, p: f64) -> Result<Tensor<CudaRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        sem_impl(self, x)
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
    fn test_describe_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let data = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let stats = client.describe(&data).unwrap();

        assert_eq!(stats.nobs, 5);

        let mean_val = extract_scalar(&stats.mean).unwrap();
        assert!((mean_val - 3.0).abs() < 1e-10);
    }
}
