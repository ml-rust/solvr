//! WebGPU implementation of K-Means clustering.

use crate::cluster::impl_generic::{kmeans_impl, kmeans_predict_impl};
use crate::cluster::traits::kmeans::{KMeansAlgorithms, KMeansOptions, KMeansResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl KMeansAlgorithms<WgpuRuntime> for WgpuClient {
    fn kmeans(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &KMeansOptions<WgpuRuntime>,
    ) -> Result<KMeansResult<WgpuRuntime>> {
        kmeans_impl(self, data, options)
    }

    fn kmeans_predict(
        &self,
        centroids: &Tensor<WgpuRuntime>,
        data: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        kmeans_predict_impl(self, centroids, data)
    }
}
