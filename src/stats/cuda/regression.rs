//! CUDA implementation of regression analysis algorithms.

use crate::stats::LinregressResult;
use crate::stats::impl_generic::linregress_impl;
use crate::stats::traits::RegressionAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl RegressionAlgorithms<CudaRuntime> for CudaClient {
    fn linregress(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        // Skip if no CUDA device available
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_linregress_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let x = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let y = Tensor::<CudaRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

        let result = client.linregress(&x, &y).unwrap();

        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
    }
}
