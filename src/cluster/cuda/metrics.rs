//! CUDA implementation of cluster evaluation metrics.

use crate::cluster::impl_generic::{
    adjusted_rand_score_impl, calinski_harabasz_score_impl, davies_bouldin_score_impl,
    homogeneity_completeness_v_measure_impl, normalized_mutual_info_score_impl,
    silhouette_samples_impl, silhouette_score_impl,
};
use crate::cluster::traits::metrics::{ClusterMetricsAlgorithms, HCVScore};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ClusterMetricsAlgorithms<CudaRuntime> for CudaClient {
    fn silhouette_score(
        &self,
        data: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        silhouette_score_impl(self, data, labels, metric)
    }

    fn silhouette_samples(
        &self,
        data: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        silhouette_samples_impl(self, data, labels, metric)
    }

    fn calinski_harabasz_score(
        &self,
        data: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        calinski_harabasz_score_impl(self, data, labels)
    }

    fn davies_bouldin_score(
        &self,
        data: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        davies_bouldin_score_impl(self, data, labels)
    }

    fn adjusted_rand_score(
        &self,
        labels_true: &Tensor<CudaRuntime>,
        labels_pred: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        adjusted_rand_score_impl(self, labels_true, labels_pred)
    }

    fn normalized_mutual_info_score(
        &self,
        labels_true: &Tensor<CudaRuntime>,
        labels_pred: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        normalized_mutual_info_score_impl(self, labels_true, labels_pred)
    }

    fn homogeneity_completeness_v_measure(
        &self,
        labels_true: &Tensor<CudaRuntime>,
        labels_pred: &Tensor<CudaRuntime>,
    ) -> Result<HCVScore<CudaRuntime>> {
        homogeneity_completeness_v_measure_impl(self, labels_true, labels_pred)
    }
}
