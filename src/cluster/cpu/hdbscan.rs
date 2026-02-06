//! CPU implementation of HDBSCAN clustering.

use crate::cluster::impl_generic::hdbscan_impl;
use crate::cluster::traits::hdbscan::{HdbscanAlgorithms, HdbscanOptions, HdbscanResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl HdbscanAlgorithms<CpuRuntime> for CpuClient {
    fn hdbscan(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &HdbscanOptions,
    ) -> Result<HdbscanResult<CpuRuntime>> {
        hdbscan_impl(self, data, options)
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
    fn test_hdbscan_basic() {
        let (client, device) = setup();

        // Two well-separated clusters with enough points for min_cluster_size
        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                0.0, 0.2,
                0.1, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
                10.0, 10.2,
                10.1, 10.0,
            ],
            &[10, 2],
            &device,
        );

        let options = HdbscanOptions {
            min_cluster_size: 3,
            min_samples: Some(2),
            ..Default::default()
        };

        let result = client.hdbscan(&data, &options).unwrap();
        assert_eq!(result.labels.shape(), &[10]);
        assert_eq!(result.probabilities.shape(), &[10]);
    }

    #[test]
    fn test_hdbscan_cluster_separation() {
        let (client, device) = setup();

        // Two clearly separated clusters — first 5 near origin, last 5 near (100,100)
        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.0,
                100.0, 100.0, 100.1, 100.1, 100.2, 100.0, 100.0, 100.2, 100.1, 100.0,
            ],
            &[10, 2],
            &device,
        );

        let options = HdbscanOptions {
            min_cluster_size: 3,
            min_samples: Some(2),
            ..Default::default()
        };

        let result = client.hdbscan(&data, &options).unwrap();
        let labels: Vec<f64> = result.labels.to_vec();
        // Points in same group should have same label
        let first_group: Vec<f64> = labels[0..5].to_vec();
        let second_group: Vec<f64> = labels[5..10].to_vec();
        // All in first group should match
        assert!(first_group.iter().all(|&l| l == first_group[0]));
        // All in second group should match
        assert!(second_group.iter().all(|&l| l == second_group[0]));
        // Two groups should differ (unless both noise)
        if first_group[0] >= 0.0 && second_group[0] >= 0.0 {
            assert_ne!(first_group[0], second_group[0]);
        }
    }

    #[test]
    fn test_hdbscan_noise_detection() {
        let (client, device) = setup();

        // Dense cluster + isolated outlier
        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.0,
                50.0, 50.0,  // outlier
            ],
            &[6, 2],
            &device,
        );

        let options = HdbscanOptions {
            min_cluster_size: 3,
            min_samples: Some(2),
            ..Default::default()
        };

        let result = client.hdbscan(&data, &options).unwrap();
        let labels: Vec<f64> = result.labels.to_vec();
        // Outlier should be noise (-1) or at least different from the cluster
        let cluster_label = labels[0];
        if cluster_label >= 0.0 {
            // The outlier should be noise
            assert!(labels[5] < 0.0 || labels[5] != cluster_label);
        }
    }
}
