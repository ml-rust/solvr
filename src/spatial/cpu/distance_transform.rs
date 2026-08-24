//! CPU implementation of distance transform algorithms.

use crate::spatial::impl_generic::distance_transform::{
    distance_transform_edt_impl, distance_transform_impl,
};
use crate::spatial::traits::distance_transform::{
    DistanceTransformAlgorithms, DistanceTransformMetric,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DistanceTransformAlgorithms<CpuRuntime> for CpuClient {
    fn distance_transform(
        &self,
        input: &Tensor<CpuRuntime>,
        metric: DistanceTransformMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        distance_transform_impl(self, input, metric)
    }

    fn distance_transform_edt(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        distance_transform_edt_impl(self, input)
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
    fn test_edt_1d() {
        let (client, device) = setup();
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client.distance_transform_edt(&input).unwrap();
        let out: Vec<f64> = result.to_vec();
        // Distances from the single foreground pixel at index 2
        assert!((out[0] - 2.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 0.0).abs() < 1e-10);
        assert!((out[3] - 1.0).abs() < 1e-10);
        assert!((out[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_edt_2d() {
        let (client, device) = setup();
        // Single foreground pixel in center
        let data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_slice(&data, &[3, 3], &device).unwrap();
        let result = client.distance_transform_edt(&input).unwrap();
        let out: Vec<f64> = result.to_vec();
        assert!((out[4] - 0.0).abs() < 1e-10); // center
        assert!((out[1] - 1.0).abs() < 1e-10); // adjacent
        assert!((out[3] - 1.0).abs() < 1e-10); // adjacent
        // Diagonal should be sqrt(2)
        assert!((out[0] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_cityblock_distance() {
        let (client, device) = setup();
        let data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_slice(&data, &[3, 3], &device).unwrap();
        let result = client
            .distance_transform(&input, DistanceTransformMetric::CityBlock)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        assert!((out[4] - 0.0).abs() < 1e-10); // center
        assert!((out[1] - 1.0).abs() < 1e-10); // adjacent
        // Diagonal = 2 (manhattan)
        assert!((out[0] - 2.0).abs() < 1e-10);
    }
}
