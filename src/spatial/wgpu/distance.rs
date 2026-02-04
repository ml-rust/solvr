//! WebGPU implementation of distance algorithms.

use crate::spatial::impl_generic::{
    cdist_impl, pdist_impl, squareform_impl, squareform_inverse_impl,
};
use crate::spatial::traits::distance::{DistanceAlgorithms, DistanceMetric};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl DistanceAlgorithms<WgpuRuntime> for WgpuClient {
    fn cdist(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        cdist_impl(self, x, y, metric)
    }

    fn pdist(
        &self,
        x: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        pdist_impl(self, x, metric)
    }

    fn squareform(&self, condensed: &Tensor<WgpuRuntime>, n: usize) -> Result<Tensor<WgpuRuntime>> {
        squareform_impl(self, condensed, n)
    }

    fn squareform_inverse(&self, square: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        squareform_inverse_impl(self, square)
    }
}
