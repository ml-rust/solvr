//! CPU implementation of spatial algorithms.
//!
//! This module implements the spatial algorithm traits for CPU
//! by delegating to the generic implementations in `impl_generic/`.

mod balltree;
mod convex_hull;
mod delaunay;
mod distance;
mod kdtree;
mod procrustes;
mod rotation;
mod voronoi;
