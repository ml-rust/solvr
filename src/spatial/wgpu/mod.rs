//! WebGPU implementation of spatial algorithms.
//!
//! This module implements the spatial algorithm traits for WebGPU
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! # Limitations
//!
//! - Only F32 is supported (WGSL doesn't support F64)

mod balltree;
mod convex_hull;
mod delaunay;
mod distance;
mod kdtree;
mod procrustes;
mod rotation;
mod voronoi;
