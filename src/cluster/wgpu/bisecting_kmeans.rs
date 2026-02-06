//! WebGPU implementation of Bisecting K-Means clustering.

use crate::cluster::impl_generic::bisecting_kmeans_impl;
use crate::cluster::traits::bisecting_kmeans::{BisectingKMeansAlgorithms, BisectingKMeansOptions};
use crate::cluster::traits::kmeans::KMeansResult;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BisectingKMeansAlgorithms<WgpuRuntime> for WgpuClient {
    fn bisecting_kmeans(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &BisectingKMeansOptions,
    ) -> Result<KMeansResult<WgpuRuntime>> {
        bisecting_kmeans_impl(self, data, options)
    }
}
