//! Spatial algorithms and data structures.
//!
//! This module provides spatial computing operations including:
//! - Distance computations (cdist, pdist, squareform)
//! - Spatial trees (KDTree, BallTree)
//! - Computational geometry (ConvexHull, Delaunay, Voronoi)
//! - Transforms (Rotation, Procrustes)
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! spatial/
//! ├── mod.rs                # Exports only
//! ├── validation.rs         # Input validation helpers
//! ├── traits/               # Algorithm trait definitions
//! │   ├── distance.rs
//! │   ├── kdtree.rs
//! │   ├── balltree.rs
//! │   ├── convex_hull.rs
//! │   ├── delaunay.rs
//! │   ├── voronoi.rs
//! │   ├── rotation.rs
//! │   └── procrustes.rs
//! ├── impl_generic/         # Generic implementations (written once)
//! │   ├── distance.rs
//! │   ├── kdtree.rs
//! │   ├── balltree.rs
//! │   ├── convex_hull.rs
//! │   ├── delaunay.rs
//! │   ├── voronoi.rs
//! │   ├── rotation.rs
//! │   └── procrustes.rs
//! ├── cpu/                  # CPU trait impl (pure delegation)
//! │   └── ...
//! ├── cuda/                 # CUDA trait impl (pure delegation)
//! │   └── ...
//! └── wgpu/                 # WebGPU trait impl (pure delegation)
//!     └── ...
//! ```
//!
//! # Backend Support
//!
//! - CPU (F32, F64)
//! - CUDA (F32, F64) - requires `cuda` feature
//! - WebGPU (F32 only) - requires `wgpu` feature

mod cpu;
pub mod impl_generic;
pub mod traits;
mod validation;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

// Re-export validation helpers
pub use validation::{
    validate_k, validate_matching_dims, validate_points_2d, validate_points_dtype, validate_radius,
};

// Re-export traits and types
pub use traits::balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
pub use traits::convex_hull::{ConvexHull, ConvexHullAlgorithms};
pub use traits::delaunay::{Delaunay, DelaunayAlgorithms};
pub use traits::distance::{DistanceAlgorithms, DistanceMetric};
pub use traits::distance_transform::{DistanceTransformAlgorithms, DistanceTransformMetric};
pub use traits::kdtree::{KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult};
pub use traits::mesh::{Mesh, MeshAlgorithms, SimplificationMethod, SmoothingMethod};
pub use traits::procrustes::{ProcrustesAlgorithms, ProcrustesResult};
pub use traits::rotation::{EulerOrder, Rotation, RotationAlgorithms};
pub use traits::voronoi::{Voronoi, VoronoiAlgorithms};
