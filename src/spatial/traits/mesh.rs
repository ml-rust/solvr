//! Mesh processing trait definitions.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A triangle mesh.
#[derive(Debug, Clone)]
pub struct Mesh<R: Runtime> {
    /// Vertex positions, shape [n_vertices, n_dims].
    pub vertices: Tensor<R>,
    /// Triangle indices, shape [n_triangles, 3] (I64 dtype).
    pub triangles: Tensor<R>,
    /// Optional per-vertex normals, shape [n_vertices, n_dims].
    pub normals: Option<Tensor<R>>,
}

/// Mesh simplification method.
#[derive(Debug, Clone, Copy)]
pub enum SimplificationMethod {
    /// Quadric Error Metrics (Garland & Heckbert).
    QuadricError,
}

/// Mesh smoothing method.
#[derive(Debug, Clone, Copy)]
pub enum SmoothingMethod {
    /// Simple Laplacian smoothing.
    Laplacian {
        /// Smoothing factor in (0, 1).
        lambda: f64,
    },
    /// Taubin smoothing (shrink-free).
    Taubin {
        /// Smoothing factor (positive).
        lambda: f64,
        /// Inflation factor (negative, |mu| > lambda).
        mu: f64,
    },
}

/// Mesh processing algorithms.
pub trait MeshAlgorithms<R: Runtime> {
    /// Triangulate a simple polygon (ear clipping algorithm).
    ///
    /// # Arguments
    /// * `vertices` - Ordered polygon vertices, shape [n_vertices, 2] or [n_vertices, 3]
    ///
    /// # Returns
    /// A Mesh with the input vertices and computed triangle indices.
    fn triangulate_polygon(&self, vertices: &Tensor<R>) -> Result<Mesh<R>>;

    /// Simplify a mesh by reducing the number of faces.
    ///
    /// # Arguments
    /// * `mesh` - Input triangle mesh
    /// * `target_faces` - Target number of faces after simplification
    /// * `method` - Simplification algorithm to use
    fn mesh_simplify(
        &self,
        mesh: &Mesh<R>,
        target_faces: usize,
        method: SimplificationMethod,
    ) -> Result<Mesh<R>>;

    /// Smooth a mesh by iteratively adjusting vertex positions.
    ///
    /// # Arguments
    /// * `mesh` - Input triangle mesh
    /// * `iterations` - Number of smoothing iterations
    /// * `method` - Smoothing algorithm to use
    fn mesh_smooth(
        &self,
        mesh: &Mesh<R>,
        iterations: usize,
        method: SmoothingMethod,
    ) -> Result<Mesh<R>>;
}
