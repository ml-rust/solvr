//! WebGPU implementation of regression analysis algorithms.

use crate::stats::LinregressResult;
use crate::stats::impl_generic::linregress_impl;
use crate::stats::traits::RegressionAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl RegressionAlgorithms<WgpuRuntime> for WgpuClient {
    fn linregress(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_linregress_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let x = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let y = Tensor::<WgpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

        let result = client.linregress(&x, &y).unwrap();

        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
    }
}
