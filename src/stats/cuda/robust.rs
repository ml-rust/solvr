//! CUDA implementation of robust statistics algorithms.

use crate::stats::impl_generic::{
    median_abs_deviation_impl, siegelslopes_impl, theilslopes_impl, trim_mean_impl,
    winsorized_mean_impl,
};
use crate::stats::traits::{RobustRegressionResult, RobustStatisticsAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl RobustStatisticsAlgorithms<CudaRuntime> for CudaClient {
    fn trim_mean(
        &self,
        x: &Tensor<CudaRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        trim_mean_impl(self, x, proportiontocut)
    }

    fn winsorized_mean(
        &self,
        x: &Tensor<CudaRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        winsorized_mean_impl(self, x, proportiontocut)
    }

    fn median_abs_deviation(
        &self,
        x: &Tensor<CudaRuntime>,
        scale: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        median_abs_deviation_impl(self, x, scale)
    }

    fn siegelslopes(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<RobustRegressionResult<CudaRuntime>> {
        siegelslopes_impl(self, x, y)
    }

    fn theilslopes(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<RobustRegressionResult<CudaRuntime>> {
        theilslopes_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_trim_mean_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let data = Tensor::<CudaRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &device,
        )
        .unwrap();
        let result = client.trim_mean(&data, 0.2).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 5.5).abs() < 1e-10);
    }
}
