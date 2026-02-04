//! Delaunay triangulation trait.
//!
//! Computes the Delaunay triangulation of a point set. The Delaunay triangulation
//! maximizes the minimum angle of all triangles and has important properties
//! for interpolation and mesh generation.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Delaunay triangulation result.
#[derive(Debug, Clone)]
pub struct Delaunay<R: Runtime> {
    /// Original points [n, d].
    pub points: Tensor<R>,

    /// Simplices of the triangulation [n_simplices, d+1] (I64 dtype).
    /// For 2D: triangles as triples of vertex indices.
    /// For 3D: tetrahedra as 4-tuples of vertex indices.
    pub simplices: Tensor<R>,

    /// Neighboring simplex indices [n_simplices, d+1] (I64 dtype).
    /// neighbors[i, j] is the simplex sharing the facet opposite vertex j.
    /// -1 indicates no neighbor (boundary).
    pub neighbors: Tensor<R>,

    /// For each point, index of one simplex containing it [n] (I64 dtype).
    /// Used for efficient point location queries.
    pub vertex_to_simplex: Tensor<R>,

    /// Vertex indices of the convex hull [n_hull] (I64 dtype).
    pub convex_hull: Tensor<R>,
}

/// Algorithmic contract for Delaunay triangulation operations.
///
/// All backends implementing Delaunay algorithms MUST implement this trait.
pub trait DelaunayAlgorithms<R: Runtime> {
    /// Compute the Delaunay triangulation of a point set.
    ///
    /// # Arguments
    ///
    /// * `points` - Point set with shape (n, d) where n >= d+1
    ///
    /// # Returns
    ///
    /// Delaunay structure containing simplices and neighbor information.
    ///
    /// # Algorithm
    ///
    /// Uses the Bowyer-Watson incremental algorithm:
    /// 1. Create a super-triangle/tetrahedron containing all points
    /// 2. Insert points one by one
    /// 3. For each point, find all simplices whose circumsphere contains it
    /// 4. Remove those simplices and re-triangulate the cavity
    /// 5. Remove simplices connected to the super-simplex
    ///
    /// Complexity: O(n^(d/2+1)) worst case, O(n log n) for well-distributed points.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 2D points
    /// let points = Tensor::from_slice(&[
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.5, 1.0,
    ///     0.5, 0.3,
    /// ], &[4, 2], &device);
    ///
    /// let tri = client.delaunay(&points)?;
    /// // tri.simplices contains triangles (3-tuples of vertex indices)
    /// ```
    fn delaunay(&self, points: &Tensor<R>) -> Result<Delaunay<R>>;

    /// Find the simplex containing each query point.
    ///
    /// # Arguments
    ///
    /// * `tri` - The Delaunay triangulation
    /// * `query` - Query points with shape (m, d)
    ///
    /// # Returns
    ///
    /// Tensor [m] (I64 dtype) with simplex indices. -1 for points outside the hull.
    fn delaunay_find_simplex(&self, tri: &Delaunay<R>, query: &Tensor<R>) -> Result<Tensor<R>>;

    /// Get the vertex neighbors of each point.
    ///
    /// # Arguments
    ///
    /// * `tri` - The Delaunay triangulation
    ///
    /// # Returns
    ///
    /// For each vertex, the indices of neighboring vertices.
    /// Returns (indices, indptr) in CSR-like format:
    /// - indices: concatenated neighbor indices
    /// - indptr: start/end positions for each vertex [n+1]
    fn delaunay_vertex_neighbors(&self, tri: &Delaunay<R>) -> Result<(Tensor<R>, Tensor<R>)>;
}

#[cfg(test)]
mod tests {
    // Tests will be in the implementation files
}
