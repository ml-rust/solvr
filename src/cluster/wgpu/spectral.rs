//! WebGPU implementation of spectral clustering.

use crate::cluster::impl_generic::spectral_clustering_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::spectral::{SpectralClusteringAlgorithms, SpectralOptions};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SpectralClusteringAlgorithms<WgpuRuntime> for WgpuClient {
    fn spectral_clustering(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &SpectralOptions,
    ) -> Result<KMeansResult<WgpuRuntime>> {
        spectral_clustering_impl(self, data, options)
    }
}
