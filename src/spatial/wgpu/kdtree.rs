//! WebGPU implementation of KDTree algorithms.

use crate::spatial::impl_generic::{
    kdtree_build_impl, kdtree_query_impl, kdtree_query_radius_impl,
};
use crate::spatial::traits::kdtree::{
    KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl KDTreeAlgorithms<WgpuRuntime> for WgpuClient {
    fn kdtree_build(
        &self,
        points: &Tensor<WgpuRuntime>,
        options: KDTreeOptions,
    ) -> Result<KDTree<WgpuRuntime>> {
        kdtree_build_impl(self, points, options)
    }

    fn kdtree_query(
        &self,
        tree: &KDTree<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
        k: usize,
    ) -> Result<KNNResult<WgpuRuntime>> {
        kdtree_query_impl(self, tree, query, k)
    }

    fn kdtree_query_radius(
        &self,
        tree: &KDTree<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<WgpuRuntime>> {
        kdtree_query_radius_impl(self, tree, query, radius)
    }
}
