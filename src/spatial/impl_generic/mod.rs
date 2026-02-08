//! Generic spatial algorithm implementations.
//!
//! This module provides Runtime-generic implementations of spatial algorithms.
//! All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! All spatial operations are fully tensor-based - data stays on device
//! with no GPU->CPU->GPU roundtrips in algorithm loops. Operations use numr's
//! tensor ops: distance computations, sorting, indexing, linear algebra, etc.
//!
//! The key benefit: **zero code duplication** across backends. CPU, CUDA, and
//! WebGPU all use these same implementations.

mod balltree;
mod convex_hull;
mod delaunay;
mod distance;
pub mod distance_transform;
mod kdtree;
pub mod mesh;
mod procrustes;
mod rotation;
mod voronoi;

// Re-export only what backends need
pub use balltree::{balltree_build_impl, balltree_query_impl, balltree_query_radius_impl};
pub use convex_hull::{convex_hull_contains_impl, convex_hull_impl};
pub use delaunay::{delaunay_find_simplex_impl, delaunay_impl, delaunay_vertex_neighbors_impl};
pub use distance::{cdist_impl, pdist_impl, squareform_impl, squareform_inverse_impl};
pub use distance_transform::{distance_transform_edt_impl, distance_transform_impl};
pub use kdtree::{kdtree_build_impl, kdtree_query_impl, kdtree_query_radius_impl};
pub use mesh::{mesh_simplify_impl, mesh_smooth_impl, triangulate_polygon_impl};
pub use procrustes::{orthogonal_procrustes_impl, procrustes_impl};
pub use rotation::{
    rotation_apply_impl, rotation_as_euler_impl, rotation_as_matrix_impl, rotation_as_quat_impl,
    rotation_as_rotvec_impl, rotation_compose_impl, rotation_from_axis_angle_impl,
    rotation_from_euler_impl, rotation_from_matrix_impl, rotation_from_quat_impl,
    rotation_from_rotvec_impl, rotation_identity_impl, rotation_inverse_impl,
    rotation_magnitude_impl, rotation_random_impl, rotation_slerp_impl,
};
pub use voronoi::{voronoi_find_region_impl, voronoi_from_delaunay_impl, voronoi_impl};
