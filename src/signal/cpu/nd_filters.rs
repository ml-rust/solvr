//! CPU implementation of N-dimensional filter algorithms.

use crate::signal::impl_generic::{
    gaussian_filter_impl, maximum_filter_impl, minimum_filter_impl, percentile_filter_impl,
    uniform_filter_impl,
};
use crate::signal::traits::nd_filters::{BoundaryMode, NdFilterAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl NdFilterAlgorithms<CpuRuntime> for CpuClient {
    fn gaussian_filter(
        &self,
        input: &Tensor<CpuRuntime>,
        sigma: &[f64],
        order: &[usize],
        mode: BoundaryMode,
        truncate: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        gaussian_filter_impl(self, input, sigma, order, mode, truncate)
    }

    fn uniform_filter(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<CpuRuntime>> {
        uniform_filter_impl(self, input, size, mode)
    }

    fn minimum_filter(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<CpuRuntime>> {
        minimum_filter_impl(self, input, size, mode)
    }

    fn maximum_filter(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<CpuRuntime>> {
        maximum_filter_impl(self, input, size, mode)
    }

    fn percentile_filter(
        &self,
        input: &Tensor<CpuRuntime>,
        percentile: f64,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<CpuRuntime>> {
        percentile_filter_impl(self, input, percentile, size, mode)
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
    fn test_gaussian_filter_1d() {
        let (client, device) = setup();
        let data = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_slice(&data, &[7], &device).unwrap();
        let result = client
            .gaussian_filter(&input, &[1.0], &[], BoundaryMode::Constant(0.0), 4.0)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Peak should be at center, values should be symmetric and sum ~1
        assert!(out[3] > out[2]);
        assert!((out[2] - out[4]).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_filter_2d() {
        let (client, device) = setup();
        // 5x5 with center spike
        let mut data = vec![0.0; 25];
        data[12] = 1.0;
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client
            .gaussian_filter(&input, &[1.0], &[], BoundaryMode::Constant(0.0), 4.0)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Center should still be the maximum
        assert!(out[12] > out[0]);
        // Should be symmetric
        assert!((out[11] - out[13]).abs() < 1e-10);
        assert!((out[7] - out[17]).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_filter_1d() {
        let (client, device) = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client
            .uniform_filter(&input, &[3], BoundaryMode::Reflect)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Middle element: (2+3+4)/3 = 3.0
        assert!((out[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_filter_1d() {
        let (client, device) = setup();
        let data = vec![5.0, 3.0, 7.0, 1.0, 4.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client
            .minimum_filter(&input, &[3], BoundaryMode::Nearest)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Position 2: min(3, 7, 1) = 1.0
        assert!((out[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximum_filter_1d() {
        let (client, device) = setup();
        let data = vec![5.0, 3.0, 7.0, 1.0, 4.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client
            .maximum_filter(&input, &[3], BoundaryMode::Nearest)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Position 2: max(3, 7, 1) = 7.0
        assert!((out[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_filter_median() {
        let (client, device) = setup();
        let data = vec![5.0, 3.0, 7.0, 1.0, 4.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client
            .percentile_filter(&input, 50.0, &[3], BoundaryMode::Nearest)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Position 2: median(3, 7, 1) = 3.0
        assert!((out[2] - 3.0).abs() < 1e-10);
    }
}
