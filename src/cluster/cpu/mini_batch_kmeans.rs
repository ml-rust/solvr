//! CPU implementation of Mini-Batch K-Means clustering.

use crate::cluster::impl_generic::mini_batch_kmeans_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::mini_batch_kmeans::{
    MiniBatchKMeansAlgorithms, MiniBatchKMeansOptions,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MiniBatchKMeansAlgorithms<CpuRuntime> for CpuClient {
    fn mini_batch_kmeans(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &MiniBatchKMeansOptions<CpuRuntime>,
    ) -> Result<KMeansResult<CpuRuntime>> {
        mini_batch_kmeans_impl(self, data, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_mini_batch_kmeans_basic() {
        let device = CpuDevice::new();
        let c = CpuClient::new(device.clone());
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
            &[6, 2],
            &device,
        );
        let opts = MiniBatchKMeansOptions {
            n_clusters: 2,
            batch_size: 4,
            max_iter: 50,
            ..Default::default()
        };
        let result = c.mini_batch_kmeans(&data, &opts).unwrap();
        assert_eq!(result.centroids.shape(), &[2, 2]);
        assert_eq!(result.labels.shape(), &[6]);
        let labels: Vec<f64> = result.labels.to_vec();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
