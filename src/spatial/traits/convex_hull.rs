//! Convex hull trait.
//!
//! Computes the convex hull of a set of points. Uses Graham scan for 2D
//! and Quickhull for 3D and higher dimensions.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Convex hull result.
#[derive(Debug, Clone)]
pub struct ConvexHull<R: Runtime> {
    /// Original points [n, d].
    pub points: Tensor<R>,

    /// Vertex indices on the hull boundary [n_vertices] (I64 dtype).
    /// For 2D: ordered counterclockwise.
    /// For 3D+: unordered set of vertices on hull.
    pub vertices: Tensor<R>,

    /// Simplices (facets) of the hull [n_simplices, d] (I64 dtype).
    /// For 2D: edges as pairs of vertex indices.
    /// For 3D: triangular faces as triples of vertex indices.
    /// For nD: (n-1)-simplices as n-tuples of vertex indices.
    pub simplices: Tensor<R>,

    /// For 3D+: neighboring simplex indices [n_simplices, d] (I64 dtype).
    /// neighbors[i, j] is the simplex sharing facet opposite to vertex j.
    /// -1 indicates no neighbor (boundary).
    pub neighbors: Option<Tensor<R>>,

    /// Equations of the hyperplanes for each simplex [n_simplices, d+1].
    /// For each simplex, (A, b) where Ax + b = 0 defines the hyperplane.
    /// The first d values are the normal A, the last is the offset b.
    pub equations: Option<Tensor<R>>,

    /// Volume (2D: area, 3D: volume) of the convex hull.
    pub volume: f64,

    /// Surface area of the convex hull.
    /// For 2D: perimeter.
    /// For 3D: surface area.
    pub area: f64,
}

/// Algorithmic contract for convex hull operations.
///
/// All backends implementing convex hull algorithms MUST implement this trait.
pub trait ConvexHullAlgorithms<R: Runtime> {
    /// Compute the convex hull of a point set.
    ///
    /// # Arguments
    ///
    /// * `points` - Point set with shape (n, d) where n >= d+1
    ///
    /// # Returns
    ///
    /// ConvexHull structure containing vertices, simplices, and geometry info.
    ///
    /// # Algorithm
    ///
    /// - 2D: Graham scan - O(n log n)
    /// - 3D+: Quickhull - O(n log n) average, O(n²) worst case
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 2D point set
    /// let points = Tensor::from_slice(&[
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     1.0, 1.0,
    ///     0.0, 1.0,
    ///     0.5, 0.5,  // Interior point
    /// ], &[5, 2], &device);
    ///
    /// let hull = client.convex_hull(&points)?;
    /// // hull.vertices contains indices [0, 1, 2, 3] (the corners)
    /// ```
    fn convex_hull(&self, points: &Tensor<R>) -> Result<ConvexHull<R>>;

    /// Test if points are inside the convex hull.
    ///
    /// # Arguments
    ///
    /// * `hull` - The convex hull
    /// * `points` - Test points with shape (m, d)
    ///
    /// # Returns
    ///
    /// Boolean tensor [m] where true indicates the point is inside or on the hull.
    fn convex_hull_contains(&self, hull: &ConvexHull<R>, points: &Tensor<R>) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    // Tests will be in the implementation files
}
