//! CUDA implementation of Mini-Batch K-Means clustering.

use crate::cluster::impl_generic::mini_batch_kmeans_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::mini_batch_kmeans::{
    MiniBatchKMeansAlgorithms, MiniBatchKMeansOptions,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MiniBatchKMeansAlgorithms<CudaRuntime> for CudaClient {
    fn mini_batch_kmeans(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &MiniBatchKMeansOptions<CudaRuntime>,
    ) -> Result<KMeansResult<CudaRuntime>> {
        mini_batch_kmeans_impl(self, data, options)
    }
}
