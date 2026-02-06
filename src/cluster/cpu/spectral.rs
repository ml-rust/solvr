//! CPU implementation of spectral clustering.

use crate::cluster::impl_generic::spectral_clustering_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::spectral::{SpectralClusteringAlgorithms, SpectralOptions};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SpectralClusteringAlgorithms<CpuRuntime> for CpuClient {
    fn spectral_clustering(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &SpectralOptions,
    ) -> Result<KMeansResult<CpuRuntime>> {
        spectral_clustering_impl(self, data, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::traits::spectral::SpectralOptions;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_spectral_clustering_basic() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                0.0, 0.2,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
                10.0, 10.2,
            ],
            &[8, 2],
            &device,
        );

        let options = SpectralOptions {
            n_clusters: 2,
            n_init: 3,
            ..Default::default()
        };

        let result = client.spectral_clustering(&data, &options).unwrap();
        assert_eq!(result.labels.shape(), &[8]);
        assert_eq!(result.centroids.shape(), &[2, 2]);
    }
}
