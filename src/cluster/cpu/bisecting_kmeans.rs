//! CPU implementation of Bisecting K-Means clustering.

use crate::cluster::impl_generic::bisecting_kmeans_impl;
use crate::cluster::traits::bisecting_kmeans::{BisectingKMeansAlgorithms, BisectingKMeansOptions};
use crate::cluster::traits::kmeans::KMeansResult;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BisectingKMeansAlgorithms<CpuRuntime> for CpuClient {
    fn bisecting_kmeans(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &BisectingKMeansOptions,
    ) -> Result<KMeansResult<CpuRuntime>> {
        bisecting_kmeans_impl(self, data, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_bisecting_kmeans_basic() {
        let device = CpuDevice::new();
        let c = CpuClient::new(device.clone());
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
            &[6, 2],
            &device,
        );
        let opts = BisectingKMeansOptions {
            n_clusters: 2,
            ..Default::default()
        };
        let result = c.bisecting_kmeans(&data, &opts).unwrap();
        assert_eq!(result.centroids.shape(), &[2, 2]);
        let labels: Vec<f64> = result.labels.to_vec();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
