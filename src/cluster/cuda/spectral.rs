//! CUDA implementation of spectral clustering.

use crate::cluster::impl_generic::spectral_clustering_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::spectral::{SpectralClusteringAlgorithms, SpectralOptions};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SpectralClusteringAlgorithms<CudaRuntime> for CudaClient {
    fn spectral_clustering(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &SpectralOptions,
    ) -> Result<KMeansResult<CudaRuntime>> {
        spectral_clustering_impl(self, data, options)
    }
}
