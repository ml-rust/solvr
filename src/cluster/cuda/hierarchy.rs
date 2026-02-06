//! CUDA implementation of hierarchical clustering.

use crate::cluster::impl_generic::{
    cut_tree_impl, fcluster_impl, fclusterdata_impl, leaves_list_impl, linkage_from_data_impl,
    linkage_impl,
};
use crate::cluster::traits::hierarchy::{
    FClusterCriterion, HierarchyAlgorithms, LinkageMatrix, LinkageMethod,
};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl HierarchyAlgorithms<CudaRuntime> for CudaClient {
    fn linkage(
        &self,
        distances: &Tensor<CudaRuntime>,
        n: usize,
        method: LinkageMethod,
    ) -> Result<LinkageMatrix<CudaRuntime>> {
        linkage_impl(self, distances, n, method)
    }

    fn linkage_from_data(
        &self,
        data: &Tensor<CudaRuntime>,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<LinkageMatrix<CudaRuntime>> {
        linkage_from_data_impl(self, data, method, metric)
    }

    fn fcluster(
        &self,
        z: &LinkageMatrix<CudaRuntime>,
        criterion: FClusterCriterion,
    ) -> Result<Tensor<CudaRuntime>> {
        fcluster_impl(self, z, criterion)
    }

    fn fclusterdata(
        &self,
        data: &Tensor<CudaRuntime>,
        criterion: FClusterCriterion,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        fclusterdata_impl(self, data, criterion, method, metric)
    }

    fn leaves_list(&self, z: &LinkageMatrix<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        leaves_list_impl(self, z)
    }

    fn cut_tree(
        &self,
        z: &LinkageMatrix<CudaRuntime>,
        n_clusters: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        cut_tree_impl(self, z, n_clusters)
    }
}
