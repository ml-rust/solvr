//! CPU implementation of robust statistics algorithms.

use crate::stats::impl_generic::{
    median_abs_deviation_impl, siegelslopes_impl, theilslopes_impl, trim_mean_impl,
    winsorized_mean_impl,
};
use crate::stats::traits::{RobustRegressionResult, RobustStatisticsAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RobustStatisticsAlgorithms<CpuRuntime> for CpuClient {
    fn trim_mean(
        &self,
        x: &Tensor<CpuRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        trim_mean_impl(self, x, proportiontocut)
    }

    fn winsorized_mean(
        &self,
        x: &Tensor<CpuRuntime>,
        proportiontocut: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        winsorized_mean_impl(self, x, proportiontocut)
    }

    fn median_abs_deviation(
        &self,
        x: &Tensor<CpuRuntime>,
        scale: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        median_abs_deviation_impl(self, x, scale)
    }

    fn siegelslopes(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<RobustRegressionResult<CpuRuntime>> {
        siegelslopes_impl(self, x, y)
    }

    fn theilslopes(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<RobustRegressionResult<CpuRuntime>> {
        theilslopes_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_trim_mean() {
        let (client, device) = setup();
        // Data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let data = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &device,
        )
        .unwrap();

        // 0% trim = regular mean
        let result = client.trim_mean(&data, 0.0).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 5.5).abs() < 1e-10);

        // 20% trim: cut 2 from each end -> mean(3,4,5,6,7,8) = 5.5
        let result = client.trim_mean(&data, 0.2).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 5.5).abs() < 1e-10);

        // 10% trim: cut 1 from each end -> mean(2..9) = 5.5
        let result = client.trim_mean(&data, 0.1).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_winsorized_mean() {
        let (client, device) = setup();
        let data = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
            &[10],
            &device,
        )
        .unwrap();

        // Without winsorizing
        let regular_mean = 145.0 / 10.0; // 14.5

        // 10% winsorize: replace 1->2, 100->9 -> mean(2,2,3,4,5,6,7,8,9,9) = 5.5
        let result = client.winsorized_mean(&data, 0.1).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!(val < regular_mean); // Should be much less due to outlier removal
        assert!((val - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_abs_deviation() {
        let (client, device) = setup();
        let data =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();

        // Median = 3, |x - 3| = [2, 1, 0, 1, 2], median(|x-3|) = 1
        let result = client.median_abs_deviation(&data, false).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 1.0).abs() < 1e-10);

        // Scaled: 1.0 * 1.4826
        let result = client.median_abs_deviation(&data, true).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 1.4826).abs() < 1e-4);
    }

    #[test]
    fn test_theilslopes() {
        let (client, device) = setup();
        let x =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device)
            .unwrap();

        let result = client.theilslopes(&x, &y).unwrap();
        let slope = extract_scalar(&result.slope).unwrap();
        let intercept = extract_scalar(&result.intercept).unwrap();

        // Perfect linear: y = 2x, slope = 2, intercept = 0
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_siegelslopes() {
        let (client, device) = setup();
        let x =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device)
            .unwrap();

        let result = client.siegelslopes(&x, &y).unwrap();
        let slope = extract_scalar(&result.slope).unwrap();
        let intercept = extract_scalar(&result.intercept).unwrap();

        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 0.0).abs() < 1e-10);
    }
}
