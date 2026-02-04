//! Distance computation trait.
//!
//! This trait wraps numr's `DistanceOps` to provide a consistent interface
//! within the solvr spatial module.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export the metric enum from numr for convenience
pub use numr::ops::DistanceMetric;

/// Algorithmic contract for distance computations.
///
/// All backends implementing distance algorithms MUST implement this trait.
/// The trait delegates to numr's `DistanceOps` for the actual computation.
pub trait DistanceAlgorithms<R: Runtime> {
    /// Compute pairwise distances between two point sets.
    ///
    /// Given two sets of points X and Y, computes the distance between
    /// every pair (x_i, y_j) and returns a distance matrix.
    ///
    /// # Arguments
    ///
    /// * `x` - First point set with shape (n, d) where n is the number of points
    ///   and d is the dimensionality
    /// * `y` - Second point set with shape (m, d)
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    ///
    /// Distance matrix with shape (n, m) where element (i, j) is the distance
    /// between x[i] and y[j].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use solvr::spatial::{DistanceAlgorithms, DistanceMetric};
    ///
    /// // Points in 3D space
    /// let x = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3], &device);
    /// let y = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0, 2.0, 2.0], &[2, 3], &device);
    ///
    /// // Euclidean distances
    /// let d = client.cdist(&x, &y, DistanceMetric::Euclidean)?;
    /// // d has shape (2, 2), d[i,j] = ||x[i] - y[j]||
    /// ```
    fn cdist(&self, x: &Tensor<R>, y: &Tensor<R>, metric: DistanceMetric) -> Result<Tensor<R>>;

    /// Compute pairwise distances within a single point set (condensed form).
    ///
    /// Computes distances between all pairs of points in X and returns
    /// the upper triangle in condensed (1D) form. This is more memory
    /// efficient than the full distance matrix for symmetric distance
    /// computation.
    ///
    /// # Arguments
    ///
    /// * `x` - Point set with shape (n, d)
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    ///
    /// Condensed distance vector with shape (n*(n-1)/2,) containing the upper
    /// triangle of the distance matrix in row-major order.
    ///
    /// For n points, the condensed form stores distances as:
    /// [d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2,n-1)]
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Tensor::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device);
    ///
    /// // Condensed distances: [d(0,1), d(0,2), d(1,2)]
    /// let d = client.pdist(&x, DistanceMetric::Euclidean)?;
    /// // d has shape (3,) = n*(n-1)/2 for n=3
    /// ```
    fn pdist(&self, x: &Tensor<R>, metric: DistanceMetric) -> Result<Tensor<R>>;

    /// Convert condensed distance vector to square distance matrix.
    ///
    /// Takes a condensed distance vector (from `pdist`) and expands it to
    /// a full symmetric distance matrix with zeros on the diagonal.
    ///
    /// # Arguments
    ///
    /// * `condensed` - Condensed distance vector with shape (n*(n-1)/2,)
    /// * `n` - Number of original points
    ///
    /// # Returns
    ///
    /// Square distance matrix with shape (n, n) where:
    /// - Diagonal elements are 0
    /// - Matrix is symmetric (d[i,j] == d[j,i])
    fn squareform(&self, condensed: &Tensor<R>, n: usize) -> Result<Tensor<R>>;

    /// Convert square distance matrix to condensed form.
    ///
    /// Takes a square symmetric distance matrix and extracts the upper
    /// triangle in condensed (1D) form.
    ///
    /// # Arguments
    ///
    /// * `square` - Square distance matrix with shape (n, n)
    ///
    /// # Returns
    ///
    /// Condensed distance vector with shape (n*(n-1)/2,)
    fn squareform_inverse(&self, square: &Tensor<R>) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_name() {
        assert_eq!(DistanceMetric::Euclidean.name(), "euclidean");
        assert_eq!(DistanceMetric::Manhattan.name(), "manhattan");
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
    }
}
