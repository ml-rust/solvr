//! WebGPU implementation of descriptive statistics algorithms.

use crate::stats::TensorDescriptiveStats;
use crate::stats::impl_generic::{
    describe_impl, iqr_impl, kurtosis_impl, percentile_impl, sem_impl, skewness_impl, zscore_impl,
};
use crate::stats::traits::DescriptiveStatisticsAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl DescriptiveStatisticsAlgorithms<WgpuRuntime> for WgpuClient {
    fn describe(&self, x: &Tensor<WgpuRuntime>) -> Result<TensorDescriptiveStats<WgpuRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<WgpuRuntime>, p: f64) -> Result<Tensor<WgpuRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        sem_impl(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_describe_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let data = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let stats = client.describe(&data).unwrap();

        assert_eq!(stats.nobs, 5);

        let mean_val = extract_scalar(&stats.mean).unwrap();
        assert!((mean_val - 3.0).abs() < 1e-3);
    }
}
