//! WebGPU implementation of robust statistics algorithms.

use crate::stats::impl_generic::{
    median_abs_deviation_impl, siegelslopes_impl, theilslopes_impl, trim_mean_impl,
    winsorized_mean_impl,
};
use crate::stats::traits::{RobustRegressionResult, RobustStatisticsAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl RobustStatisticsAlgorithms<WgpuRuntime> for WgpuClient {
    fn trim_mean(
        &self,
        x: &Tensor<WgpuRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        trim_mean_impl(self, x, proportiontocut)
    }

    fn winsorized_mean(
        &self,
        x: &Tensor<WgpuRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        winsorized_mean_impl(self, x, proportiontocut)
    }

    fn median_abs_deviation(
        &self,
        x: &Tensor<WgpuRuntime>,
        scale: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        median_abs_deviation_impl(self, x, scale)
    }

    fn siegelslopes(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<RobustRegressionResult<WgpuRuntime>> {
        siegelslopes_impl(self, x, y)
    }

    fn theilslopes(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<RobustRegressionResult<WgpuRuntime>> {
        theilslopes_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_trim_mean_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let data = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &device,
        );
        let result = client.trim_mean(&data, 0.2).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 5.5).abs() < 1e-3);
    }
}
