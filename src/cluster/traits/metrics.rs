//! Cluster evaluation metrics trait.

use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Homogeneity, Completeness, V-Measure scores.
#[derive(Debug, Clone)]
pub struct HCVScore<R: Runtime> {
    pub homogeneity: Tensor<R>,
    pub completeness: Tensor<R>,
    pub v_measure: Tensor<R>,
}

/// Cluster evaluation metrics.
pub trait ClusterMetricsAlgorithms<R: Runtime> {
    /// Mean silhouette coefficient (scalar).
    fn silhouette_score(
        &self,
        data: &Tensor<R>,
        labels: &Tensor<R>,
        metric: DistanceMetric,
    ) -> Result<Tensor<R>>;

    /// Per-sample silhouette coefficients [n].
    fn silhouette_samples(
        &self,
        data: &Tensor<R>,
        labels: &Tensor<R>,
        metric: DistanceMetric,
    ) -> Result<Tensor<R>>;

    /// Calinski-Harabasz index (scalar).
    fn calinski_harabasz_score(&self, data: &Tensor<R>, labels: &Tensor<R>) -> Result<Tensor<R>>;

    /// Davies-Bouldin index (scalar).
    fn davies_bouldin_score(&self, data: &Tensor<R>, labels: &Tensor<R>) -> Result<Tensor<R>>;

    /// Adjusted Rand Index (scalar).
    fn adjusted_rand_score(
        &self,
        labels_true: &Tensor<R>,
        labels_pred: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Normalized Mutual Information (scalar).
    fn normalized_mutual_info_score(
        &self,
        labels_true: &Tensor<R>,
        labels_pred: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Homogeneity, Completeness, V-Measure.
    fn homogeneity_completeness_v_measure(
        &self,
        labels_true: &Tensor<R>,
        labels_pred: &Tensor<R>,
    ) -> Result<HCVScore<R>>;
}
