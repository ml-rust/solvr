//! Voronoi diagram trait.
//!
//! Computes the Voronoi diagram (dual of Delaunay triangulation) of a point set.
//! The Voronoi diagram partitions space into regions where each region contains
//! all points closer to one generator than any other.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::delaunay::Delaunay;

/// Voronoi diagram result.
#[derive(Debug, Clone)]
pub struct Voronoi<R: Runtime> {
    /// Generator points (original input) [n, d].
    pub points: Tensor<R>,

    /// Voronoi vertices [n_vertices, d].
    /// These are the circumcenters of the Delaunay simplices.
    pub vertices: Tensor<R>,

    /// For each Voronoi region (generator), indices of vertices forming the region.
    /// Stored in CSR-like format.
    /// - ridge_vertices: concatenated vertex indices for all ridges
    /// - ridge_points: for each ridge, the two generator points it separates [n_ridges, 2]
    pub ridge_vertices: Tensor<R>,

    /// For each ridge, the two generator points it separates [n_ridges, 2] (I64 dtype).
    pub ridge_points: Tensor<R>,

    /// For each generator, indices of ridges forming its region boundary.
    /// Stored as (indices, indptr) in CSR-like format.
    pub regions_indices: Tensor<R>,
    pub regions_indptr: Tensor<R>,

    /// Point indices with unbounded regions [n_unbounded] (I64 dtype).
    /// Regions extending to infinity.
    pub point_region: Tensor<R>,
}

/// Algorithmic contract for Voronoi diagram operations.
///
/// All backends implementing Voronoi algorithms MUST implement this trait.
pub trait VoronoiAlgorithms<R: Runtime> {
    /// Compute the Voronoi diagram of a point set.
    ///
    /// # Arguments
    ///
    /// * `points` - Generator points with shape (n, d) where n >= d+1
    ///
    /// # Returns
    ///
    /// Voronoi structure containing vertices, regions, and ridges.
    ///
    /// # Algorithm
    ///
    /// The Voronoi diagram is computed as the dual of the Delaunay triangulation:
    /// 1. Compute Delaunay triangulation
    /// 2. For each Delaunay simplex, compute circumcenter (Voronoi vertex)
    /// 3. Connect circumcenters of adjacent simplices (Voronoi edges)
    /// 4. Handle unbounded regions at the convex hull boundary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let points = Tensor::from_slice(&[
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.5, 1.0,
    /// ], &[3, 2], &device);
    ///
    /// let vor = client.voronoi(&points)?;
    /// ```
    fn voronoi(&self, points: &Tensor<R>) -> Result<Voronoi<R>>;

    /// Compute the Voronoi diagram from an existing Delaunay triangulation.
    ///
    /// More efficient when you already have the Delaunay triangulation.
    fn voronoi_from_delaunay(&self, tri: &Delaunay<R>) -> Result<Voronoi<R>>;

    /// Find which Voronoi region contains each query point.
    ///
    /// # Arguments
    ///
    /// * `vor` - The Voronoi diagram
    /// * `query` - Query points with shape (m, d)
    ///
    /// # Returns
    ///
    /// Tensor [m] (I64 dtype) with generator (region) indices.
    fn voronoi_find_region(&self, vor: &Voronoi<R>, query: &Tensor<R>) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    // Tests will be in the implementation files
}
