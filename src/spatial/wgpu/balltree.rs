//! WebGPU implementation of BallTree algorithms.

use crate::spatial::impl_generic::{
    balltree_build_impl, balltree_query_impl, balltree_query_radius_impl,
};
use crate::spatial::traits::balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
use crate::spatial::traits::kdtree::{KNNResult, RadiusResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BallTreeAlgorithms<WgpuRuntime> for WgpuClient {
    fn balltree_build(
        &self,
        points: &Tensor<WgpuRuntime>,
        options: BallTreeOptions,
    ) -> Result<BallTree<WgpuRuntime>> {
        balltree_build_impl(self, points, options)
    }

    fn balltree_query(
        &self,
        tree: &BallTree<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
        k: usize,
    ) -> Result<KNNResult<WgpuRuntime>> {
        balltree_query_impl(self, tree, query, k)
    }

    fn balltree_query_radius(
        &self,
        tree: &BallTree<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<WgpuRuntime>> {
        balltree_query_radius_impl(self, tree, query, radius)
    }
}
