//! WebGPU implementation of cluster evaluation metrics.

use crate::cluster::impl_generic::{
    adjusted_rand_score_impl, calinski_harabasz_score_impl, davies_bouldin_score_impl,
    homogeneity_completeness_v_measure_impl, normalized_mutual_info_score_impl,
    silhouette_samples_impl, silhouette_score_impl,
};
use crate::cluster::traits::metrics::{ClusterMetricsAlgorithms, HCVScore};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ClusterMetricsAlgorithms<WgpuRuntime> for WgpuClient {
    fn silhouette_score(
        &self,
        data: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        silhouette_score_impl(self, data, labels, metric)
    }

    fn silhouette_samples(
        &self,
        data: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        silhouette_samples_impl(self, data, labels, metric)
    }

    fn calinski_harabasz_score(
        &self,
        data: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        calinski_harabasz_score_impl(self, data, labels)
    }

    fn davies_bouldin_score(
        &self,
        data: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        davies_bouldin_score_impl(self, data, labels)
    }

    fn adjusted_rand_score(
        &self,
        labels_true: &Tensor<WgpuRuntime>,
        labels_pred: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        adjusted_rand_score_impl(self, labels_true, labels_pred)
    }

    fn normalized_mutual_info_score(
        &self,
        labels_true: &Tensor<WgpuRuntime>,
        labels_pred: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        normalized_mutual_info_score_impl(self, labels_true, labels_pred)
    }

    fn homogeneity_completeness_v_measure(
        &self,
        labels_true: &Tensor<WgpuRuntime>,
        labels_pred: &Tensor<WgpuRuntime>,
    ) -> Result<HCVScore<WgpuRuntime>> {
        homogeneity_completeness_v_measure_impl(self, labels_true, labels_pred)
    }
}
