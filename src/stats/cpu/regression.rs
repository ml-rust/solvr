//! CPU implementation of regression analysis algorithms.

use crate::stats::LinregressResult;
use crate::stats::impl_generic::linregress_impl;
use crate::stats::traits::RegressionAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RegressionAlgorithms<CpuRuntime> for CpuClient {
    fn linregress(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_linregress() {
        let (client, device) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

        let result = client.linregress(&x, &y).unwrap();

        // Perfect linear relationship: y = 2x
        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
        assert!((result.rvalue - 1.0).abs() < 1e-10);
    }
}
