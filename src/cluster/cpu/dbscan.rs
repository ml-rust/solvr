//! CPU implementation of DBSCAN clustering.

use crate::cluster::impl_generic::dbscan_impl;
use crate::cluster::traits::dbscan::{DbscanAlgorithms, DbscanOptions, DbscanResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DbscanAlgorithms<CpuRuntime> for CpuClient {
    fn dbscan(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &DbscanOptions,
    ) -> Result<DbscanResult<CpuRuntime>> {
        dbscan_impl(self, data, options)
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
    fn test_dbscan_two_clusters() {
        let (c, device) = setup();
        // Two well-separated clusters
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
            &[6, 2],
            &device,
        );
        let opts = DbscanOptions {
            eps: 1.0,
            min_samples: 2,
            ..Default::default()
        };
        let result = c.dbscan(&data, &opts).unwrap();
        assert_eq!(result.n_clusters, 2);
        let labels: Vec<f64> = result.labels.to_vec();
        // First 3 should be same cluster, last 3 same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_dbscan_noise() {
        let (c, device) = setup();
        // 3 close points + 1 outlier
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0],
            &[4, 2],
            &device,
        );
        let opts = DbscanOptions {
            eps: 0.5,
            min_samples: 2,
            ..Default::default()
        };
        let result = c.dbscan(&data, &opts).unwrap();
        let labels: Vec<f64> = result.labels.to_vec();
        // Outlier should be noise (-1.0)
        assert_eq!(labels[3], -1.0);
    }
}
