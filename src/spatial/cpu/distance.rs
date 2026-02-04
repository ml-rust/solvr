//! CPU implementation of distance algorithms.

use crate::spatial::impl_generic::{
    cdist_impl, pdist_impl, squareform_impl, squareform_inverse_impl,
};
use crate::spatial::traits::distance::{DistanceAlgorithms, DistanceMetric};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DistanceAlgorithms<CpuRuntime> for CpuClient {
    fn cdist(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        cdist_impl(self, x, y, metric)
    }

    fn pdist(&self, x: &Tensor<CpuRuntime>, metric: DistanceMetric) -> Result<Tensor<CpuRuntime>> {
        pdist_impl(self, x, metric)
    }

    fn squareform(&self, condensed: &Tensor<CpuRuntime>, n: usize) -> Result<Tensor<CpuRuntime>> {
        squareform_impl(self, condensed, n)
    }

    fn squareform_inverse(&self, square: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        squareform_inverse_impl(self, square)
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
    fn test_cdist_euclidean() {
        let (client, device) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 2.0, 2.0], &[2, 2], &device);

        let result = client.cdist(&x, &y, DistanceMetric::Euclidean).unwrap();

        assert_eq!(result.shape(), &[3, 2]);

        let data: Vec<f64> = result.to_vec();
        // Distance from (0,0) to (1,1) = sqrt(2)
        assert!((data[0] - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_pdist_euclidean() {
        let (client, device) = setup();

        // Triangle at (0,0), (1,0), (0,1)
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device);

        let result = client.pdist(&x, DistanceMetric::Euclidean).unwrap();

        // Condensed form: [d(0,1), d(0,2), d(1,2)] = [1, 1, sqrt(2)]
        assert_eq!(result.shape(), &[3]);

        let data: Vec<f64> = result.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6); // (0,0) to (1,0)
        assert!((data[1] - 1.0).abs() < 1e-6); // (0,0) to (0,1)
        assert!((data[2] - std::f64::consts::SQRT_2).abs() < 1e-6); // (1,0) to (0,1)
    }

    #[test]
    fn test_squareform_roundtrip() {
        let (client, device) = setup();

        let condensed = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);

        let square = client.squareform(&condensed, 3).unwrap();
        assert_eq!(square.shape(), &[3, 3]);

        let back = client.squareform_inverse(&square).unwrap();
        assert_eq!(back.shape(), &[3]);

        let back_data: Vec<f64> = back.to_vec();
        assert!((back_data[0] - 1.0).abs() < 1e-6);
        assert!((back_data[1] - 2.0).abs() < 1e-6);
        assert!((back_data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cdist_manhattan() {
        let (client, device) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[1, 2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[1, 2], &device);

        let result = client.cdist(&x, &y, DistanceMetric::Manhattan).unwrap();

        let data: Vec<f64> = result.to_vec();
        // |0-1| + |0-2| = 3
        assert!((data[0] - 3.0).abs() < 1e-6);
    }
}
