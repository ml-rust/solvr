//! CUDA implementation of BallTree algorithms.

use crate::spatial::impl_generic::{
    balltree_build_impl, balltree_query_impl, balltree_query_radius_impl,
};
use crate::spatial::traits::balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
use crate::spatial::traits::kdtree::{KNNResult, RadiusResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BallTreeAlgorithms<CudaRuntime> for CudaClient {
    fn balltree_build(
        &self,
        points: &Tensor<CudaRuntime>,
        options: BallTreeOptions,
    ) -> Result<BallTree<CudaRuntime>> {
        balltree_build_impl(self, points, options)
    }

    fn balltree_query(
        &self,
        tree: &BallTree<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
        k: usize,
    ) -> Result<KNNResult<CudaRuntime>> {
        balltree_query_impl(self, tree, query, k)
    }

    fn balltree_query_radius(
        &self,
        tree: &BallTree<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<CudaRuntime>> {
        balltree_query_radius_impl(self, tree, query, radius)
    }
}
