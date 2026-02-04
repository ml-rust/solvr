//! KDTree spatial index trait.
//!
//! KDTree is a space-partitioning data structure for organizing points in a
//! k-dimensional space. Enables efficient nearest neighbor queries and
//! range searches.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::distance::DistanceMetric;

/// Options for KDTree construction.
#[derive(Debug, Clone)]
pub struct KDTreeOptions {
    /// Maximum number of points per leaf node.
    /// Larger values reduce tree depth but increase leaf search time.
    /// Default: 10
    pub leaf_size: usize,

    /// Distance metric for queries.
    /// Default: Euclidean
    pub metric: DistanceMetric,
}

impl Default for KDTreeOptions {
    fn default() -> Self {
        Self {
            leaf_size: 10,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// KDTree spatial index structure.
///
/// Stores the tree structure as flat tensors for efficient GPU operations.
#[derive(Debug, Clone)]
pub struct KDTree<R: Runtime> {
    /// Original point data [n, d].
    pub data: Tensor<R>,

    /// Split dimension for each internal node [n_internal].
    pub split_dims: Tensor<R>,

    /// Split value for each internal node [n_internal].
    pub split_values: Tensor<R>,

    /// Left child indices for each node [n_nodes]. -1 for leaves.
    pub left_children: Tensor<R>,

    /// Right child indices for each node [n_nodes]. -1 for leaves.
    pub right_children: Tensor<R>,

    /// Point indices in depth-first order [n].
    /// Points for a leaf span [leaf_starts[i], leaf_starts[i] + leaf_sizes[i]).
    pub point_indices: Tensor<R>,

    /// Start index in point_indices for each leaf [n_leaves].
    pub leaf_starts: Tensor<R>,

    /// Number of points in each leaf [n_leaves].
    pub leaf_sizes: Tensor<R>,

    /// Tree construction options.
    pub options: KDTreeOptions,
}

/// Result of k-nearest neighbors query.
#[derive(Debug, Clone)]
pub struct KNNResult<R: Runtime> {
    /// Distances to the k nearest neighbors [n_queries, k].
    pub distances: Tensor<R>,

    /// Indices of the k nearest neighbors [n_queries, k] (I64 dtype).
    pub indices: Tensor<R>,
}

/// Result of radius search query.
#[derive(Debug, Clone)]
pub struct RadiusResult<R: Runtime> {
    /// Distances to neighbors within radius [total_neighbors].
    pub distances: Tensor<R>,

    /// Indices of neighbors within radius [total_neighbors] (I64 dtype).
    pub indices: Tensor<R>,

    /// Number of neighbors for each query point [n_queries] (I64 dtype).
    pub counts: Tensor<R>,

    /// Start index in distances/indices for each query [n_queries] (I64 dtype).
    pub offsets: Tensor<R>,
}

/// Algorithmic contract for KDTree operations.
///
/// All backends implementing KDTree algorithms MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait KDTreeAlgorithms<R: Runtime> {
    /// Build a KDTree from a point set.
    ///
    /// # Arguments
    ///
    /// * `points` - Point set with shape (n, d)
    /// * `options` - Tree construction options
    ///
    /// # Returns
    ///
    /// A KDTree structure ready for queries.
    fn kdtree_build(&self, points: &Tensor<R>, options: KDTreeOptions) -> Result<KDTree<R>>;

    /// Find the k nearest neighbors for each query point.
    ///
    /// # Arguments
    ///
    /// * `tree` - The KDTree to query
    /// * `query` - Query points with shape (m, d)
    /// * `k` - Number of neighbors to find
    ///
    /// # Returns
    ///
    /// KNNResult containing distances and indices of the k nearest neighbors
    /// for each query point.
    fn kdtree_query(&self, tree: &KDTree<R>, query: &Tensor<R>, k: usize) -> Result<KNNResult<R>>;

    /// Find all neighbors within a given radius for each query point.
    ///
    /// # Arguments
    ///
    /// * `tree` - The KDTree to query
    /// * `query` - Query points with shape (m, d)
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// RadiusResult containing distances, indices, and counts for each query.
    fn kdtree_query_radius(
        &self,
        tree: &KDTree<R>,
        query: &Tensor<R>,
        radius: f64,
    ) -> Result<RadiusResult<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_options_default() {
        let opts = KDTreeOptions::default();
        assert_eq!(opts.leaf_size, 10);
        assert_eq!(opts.metric, DistanceMetric::Euclidean);
    }
}
