//! Hierarchical clustering trait.

use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkageMethod {
    /// Minimum distance between clusters.
    Single,
    /// Maximum distance between clusters.
    Complete,
    /// Average distance between all pairs.
    #[default]
    Average,
    /// Weighted average (WPGMA).
    Weighted,
    /// Distance between centroids (requires Euclidean).
    Centroid,
    /// Weighted centroid (WPGMC).
    Median,
    /// Minimize within-cluster variance (requires Euclidean).
    Ward,
}

/// Criterion for cutting a dendrogram.
#[derive(Debug, Clone)]
pub enum FClusterCriterion {
    /// Cut at a given distance threshold.
    Distance(f64),
    /// Cut to produce exactly this many clusters.
    MaxClust(usize),
}

/// Linkage matrix [n-1, 4]: each row is [id1, id2, distance, count].
#[derive(Debug, Clone)]
pub struct LinkageMatrix<R: Runtime> {
    /// The linkage matrix tensor [n-1, 4].
    pub z: Tensor<R>,
}

/// Hierarchical (agglomerative) clustering algorithms.
pub trait HierarchyAlgorithms<R: Runtime> {
    /// Compute linkage from a condensed distance matrix.
    fn linkage(
        &self,
        distances: &Tensor<R>,
        n: usize,
        method: LinkageMethod,
    ) -> Result<LinkageMatrix<R>>;

    /// Compute linkage directly from data points [n, d].
    fn linkage_from_data(
        &self,
        data: &Tensor<R>,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<LinkageMatrix<R>>;

    /// Cut dendrogram to form flat clusters.
    fn fcluster(&self, z: &LinkageMatrix<R>, criterion: FClusterCriterion) -> Result<Tensor<R>>;

    /// Cluster data directly (linkage + fcluster).
    fn fclusterdata(
        &self,
        data: &Tensor<R>,
        criterion: FClusterCriterion,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<Tensor<R>>;

    /// Return leaves in dendrogram order.
    fn leaves_list(&self, z: &LinkageMatrix<R>) -> Result<Tensor<R>>;

    /// Cut tree at multiple levels to produce cluster assignments.
    fn cut_tree(&self, z: &LinkageMatrix<R>, n_clusters: &[usize]) -> Result<Tensor<R>>;
}
