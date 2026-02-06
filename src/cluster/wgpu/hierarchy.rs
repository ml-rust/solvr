//! WebGPU implementation of hierarchical clustering.

use crate::cluster::impl_generic::{
    cut_tree_impl, fcluster_impl, fclusterdata_impl, leaves_list_impl, linkage_from_data_impl,
    linkage_impl,
};
use crate::cluster::traits::hierarchy::{
    FClusterCriterion, HierarchyAlgorithms, LinkageMatrix, LinkageMethod,
};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl HierarchyAlgorithms<WgpuRuntime> for WgpuClient {
    fn linkage(
        &self,
        distances: &Tensor<WgpuRuntime>,
        n: usize,
        method: LinkageMethod,
    ) -> Result<LinkageMatrix<WgpuRuntime>> {
        linkage_impl(self, distances, n, method)
    }

    fn linkage_from_data(
        &self,
        data: &Tensor<WgpuRuntime>,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<LinkageMatrix<WgpuRuntime>> {
        linkage_from_data_impl(self, data, method, metric)
    }

    fn fcluster(
        &self,
        z: &LinkageMatrix<WgpuRuntime>,
        criterion: FClusterCriterion,
    ) -> Result<Tensor<WgpuRuntime>> {
        fcluster_impl(self, z, criterion)
    }

    fn fclusterdata(
        &self,
        data: &Tensor<WgpuRuntime>,
        criterion: FClusterCriterion,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        fclusterdata_impl(self, data, criterion, method, metric)
    }

    fn leaves_list(&self, z: &LinkageMatrix<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        leaves_list_impl(self, z)
    }

    fn cut_tree(
        &self,
        z: &LinkageMatrix<WgpuRuntime>,
        n_clusters: &[usize],
    ) -> Result<Tensor<WgpuRuntime>> {
        cut_tree_impl(self, z, n_clusters)
    }
}
