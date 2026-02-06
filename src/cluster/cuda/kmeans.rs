//! CUDA implementation of K-Means clustering.

use crate::cluster::impl_generic::{kmeans_impl, kmeans_predict_impl};
use crate::cluster::traits::kmeans::{KMeansAlgorithms, KMeansOptions, KMeansResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl KMeansAlgorithms<CudaRuntime> for CudaClient {
    fn kmeans(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &KMeansOptions<CudaRuntime>,
    ) -> Result<KMeansResult<CudaRuntime>> {
        kmeans_impl(self, data, options)
    }

    fn kmeans_predict(
        &self,
        centroids: &Tensor<CudaRuntime>,
        data: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        kmeans_predict_impl(self, centroids, data)
    }
}
