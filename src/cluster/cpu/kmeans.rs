//! CPU implementation of K-Means clustering.

use crate::cluster::impl_generic::{kmeans_impl, kmeans_predict_impl};
use crate::cluster::traits::kmeans::{KMeansAlgorithms, KMeansOptions, KMeansResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl KMeansAlgorithms<CpuRuntime> for CpuClient {
    fn kmeans(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &KMeansOptions<CpuRuntime>,
    ) -> Result<KMeansResult<CpuRuntime>> {
        kmeans_impl(self, data, options)
    }

    fn kmeans_predict(
        &self,
        centroids: &Tensor<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        kmeans_predict_impl(self, centroids, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::traits::kmeans::{KMeansInit, KMeansOptions};
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_kmeans_basic() {
        let (client, device) = setup();

        // Two well-separated clusters
        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let options = KMeansOptions {
            n_clusters: 2,
            max_iter: 100,
            tol: 1e-4,
            n_init: 3,
            init: KMeansInit::KMeansPlusPlus,
            ..Default::default()
        };

        let result = client.kmeans(&data, &options).unwrap();
        assert_eq!(result.centroids.shape(), &[2, 2]);
        assert_eq!(result.labels.shape(), &[6]);

        // Points in same cluster should have same label
        let labels: Vec<i64> = result.labels.to_vec();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        let (client, device) = setup();

        let data =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        let options = KMeansOptions {
            n_clusters: 3,
            n_init: 1,
            ..Default::default()
        };

        let result = client.kmeans(&data, &options).unwrap();
        assert_eq!(result.centroids.shape(), &[3, 2]);
        // Each point should be its own cluster, inertia should be ~0
        let inertia: f64 = result.inertia.item().unwrap();
        assert!(inertia < 1e-6);
    }

    #[test]
    fn test_kmeans_predict() {
        let (client, device) = setup();

        let centroids = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 10.0, 10.0], &[2, 2], &device);

        let data =
            Tensor::<CpuRuntime>::from_slice(&[0.1, 0.1, 9.9, 9.9, 0.2, -0.1], &[3, 2], &device);

        let labels = client.kmeans_predict(&centroids, &data).unwrap();
        let labels_vec: Vec<i64> = labels.to_vec();
        assert_eq!(labels_vec[0], 0);
        assert_eq!(labels_vec[1], 1);
        assert_eq!(labels_vec[2], 0);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let (client, device) = setup();

        let data =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        let options = KMeansOptions {
            n_clusters: 1,
            n_init: 1,
            ..Default::default()
        };

        let result = client.kmeans(&data, &options).unwrap();
        let labels: Vec<i64> = result.labels.to_vec();
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_kmeans_with_provided_init() {
        let (client, device) = setup();

        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            &[4, 2],
            &device,
        );

        let init_centroids =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 10.0, 10.0], &[2, 2], &device);

        let options = KMeansOptions {
            n_clusters: 2,
            init: KMeansInit::Points(init_centroids),
            ..Default::default()
        };

        let result = client.kmeans(&data, &options).unwrap();
        let labels: Vec<i64> = result.labels.to_vec();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
