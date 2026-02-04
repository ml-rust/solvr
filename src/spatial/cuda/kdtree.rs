//! CUDA implementation of KDTree algorithms.

use crate::spatial::impl_generic::{
    kdtree_build_impl, kdtree_query_impl, kdtree_query_radius_impl,
};
use crate::spatial::traits::kdtree::{
    KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl KDTreeAlgorithms<CudaRuntime> for CudaClient {
    fn kdtree_build(
        &self,
        points: &Tensor<CudaRuntime>,
        options: KDTreeOptions,
    ) -> Result<KDTree<CudaRuntime>> {
        kdtree_build_impl(self, points, options)
    }

    fn kdtree_query(
        &self,
        tree: &KDTree<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
        k: usize,
    ) -> Result<KNNResult<CudaRuntime>> {
        kdtree_query_impl(self, tree, query, k)
    }

    fn kdtree_query_radius(
        &self,
        tree: &KDTree<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<CudaRuntime>> {
        kdtree_query_radius_impl(self, tree, query, radius)
    }
}
