//! CUDA implementation of Bisecting K-Means clustering.

use crate::cluster::impl_generic::bisecting_kmeans_impl;
use crate::cluster::traits::bisecting_kmeans::{BisectingKMeansAlgorithms, BisectingKMeansOptions};
use crate::cluster::traits::kmeans::KMeansResult;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BisectingKMeansAlgorithms<CudaRuntime> for CudaClient {
    fn bisecting_kmeans(
        &self,
        data: &Tensor<CudaRuntime>,
        options: &BisectingKMeansOptions,
    ) -> Result<KMeansResult<CudaRuntime>> {
        bisecting_kmeans_impl(self, data, options)
    }
}
