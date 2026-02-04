//! BallTree spatial index trait.
//!
//! BallTree is a space-partitioning data structure that uses nested hyperspheres
//! to organize points. Unlike KDTree which uses axis-aligned splits, BallTree
//! can handle arbitrary distance metrics efficiently.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::distance::DistanceMetric;
use super::kdtree::{KNNResult, RadiusResult};

/// Options for BallTree construction.
#[derive(Debug, Clone)]
pub struct BallTreeOptions {
    /// Maximum number of points per leaf node.
    /// Default: 40
    pub leaf_size: usize,

    /// Distance metric for queries.
    /// BallTree supports arbitrary metrics, unlike KDTree.
    /// Default: Euclidean
    pub metric: DistanceMetric,
}

impl Default for BallTreeOptions {
    fn default() -> Self {
        Self {
            leaf_size: 40,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// BallTree spatial index structure.
///
/// Stores the tree structure as flat tensors for efficient GPU operations.
#[derive(Debug, Clone)]
pub struct BallTree<R: Runtime> {
    /// Original point data [n, d].
    pub data: Tensor<R>,

    /// Center of each ball [n_nodes, d].
    pub centers: Tensor<R>,

    /// Radius of each ball [n_nodes].
    pub radii: Tensor<R>,

    /// Left child indices for each node [n_nodes]. -1 for leaves.
    pub left_children: Tensor<R>,

    /// Right child indices for each node [n_nodes]. -1 for leaves.
    pub right_children: Tensor<R>,

    /// Point indices in depth-first order [n].
    pub point_indices: Tensor<R>,

    /// Start index in point_indices for each leaf [n_leaves].
    pub leaf_starts: Tensor<R>,

    /// Number of points in each leaf [n_leaves].
    pub leaf_sizes: Tensor<R>,

    /// Tree construction options.
    pub options: BallTreeOptions,
}

/// Algorithmic contract for BallTree operations.
///
/// All backends implementing BallTree algorithms MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait BallTreeAlgorithms<R: Runtime> {
    /// Build a BallTree from a point set.
    ///
    /// # Arguments
    ///
    /// * `points` - Point set with shape (n, d)
    /// * `options` - Tree construction options
    ///
    /// # Returns
    ///
    /// A BallTree structure ready for queries.
    fn balltree_build(&self, points: &Tensor<R>, options: BallTreeOptions) -> Result<BallTree<R>>;

    /// Find the k nearest neighbors for each query point.
    ///
    /// # Arguments
    ///
    /// * `tree` - The BallTree to query
    /// * `query` - Query points with shape (m, d)
    /// * `k` - Number of neighbors to find
    ///
    /// # Returns
    ///
    /// KNNResult containing distances and indices of the k nearest neighbors.
    fn balltree_query(
        &self,
        tree: &BallTree<R>,
        query: &Tensor<R>,
        k: usize,
    ) -> Result<KNNResult<R>>;

    /// Find all neighbors within a given radius for each query point.
    ///
    /// # Arguments
    ///
    /// * `tree` - The BallTree to query
    /// * `query` - Query points with shape (m, d)
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// RadiusResult containing distances, indices, and counts for each query.
    fn balltree_query_radius(
        &self,
        tree: &BallTree<R>,
        query: &Tensor<R>,
        radius: f64,
    ) -> Result<RadiusResult<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balltree_options_default() {
        let opts = BallTreeOptions::default();
        assert_eq!(opts.leaf_size, 40);
        assert_eq!(opts.metric, DistanceMetric::Euclidean);
    }
}
