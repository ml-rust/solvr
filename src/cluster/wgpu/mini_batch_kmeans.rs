//! WebGPU implementation of Mini-Batch K-Means clustering.

use crate::cluster::impl_generic::mini_batch_kmeans_impl;
use crate::cluster::traits::kmeans::KMeansResult;
use crate::cluster::traits::mini_batch_kmeans::{
    MiniBatchKMeansAlgorithms, MiniBatchKMeansOptions,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MiniBatchKMeansAlgorithms<WgpuRuntime> for WgpuClient {
    fn mini_batch_kmeans(
        &self,
        data: &Tensor<WgpuRuntime>,
        options: &MiniBatchKMeansOptions<WgpuRuntime>,
    ) -> Result<KMeansResult<WgpuRuntime>> {
        mini_batch_kmeans_impl(self, data, options)
    }
}
