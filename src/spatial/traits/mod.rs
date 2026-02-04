//! Spatial algorithm traits.
//!
//! This module defines the algorithmic contracts for spatial operations.
//! Each trait represents a logical group of related algorithms.

pub mod balltree;
pub mod convex_hull;
pub mod delaunay;
pub mod distance;
pub mod kdtree;
pub mod procrustes;
pub mod rotation;
pub mod voronoi;

pub use balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
pub use convex_hull::{ConvexHull, ConvexHullAlgorithms};
pub use delaunay::{Delaunay, DelaunayAlgorithms};
pub use distance::{DistanceAlgorithms, DistanceMetric};
pub use kdtree::{KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult};
pub use procrustes::{ProcrustesAlgorithms, ProcrustesResult};
pub use rotation::{EulerOrder, Rotation, RotationAlgorithms};
pub use voronoi::{Voronoi, VoronoiAlgorithms};
