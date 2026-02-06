//! Clustering algorithms.
//!
//! Provides partitional, hierarchical, density-based, and model-based clustering
//! with full backend support (CPU, CUDA, WebGPU).

mod cpu;
pub mod impl_generic;
pub mod traits;
mod validation;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::affinity_propagation::{
    AffinityPropagationAlgorithms, AffinityPropagationOptions, AffinityPropagationResult,
};
pub use traits::bisecting_kmeans::{
    BisectingKMeansAlgorithms, BisectingKMeansOptions, BisectingStrategy,
};
pub use traits::dbscan::{DbscanAlgorithms, DbscanOptions, DbscanResult};
pub use traits::gmm::{CovarianceType, GmmAlgorithms, GmmInit, GmmModel, GmmOptions};
pub use traits::hdbscan::{
    ClusterSelectionMethod, HdbscanAlgorithms, HdbscanOptions, HdbscanResult,
};
pub use traits::hierarchy::{FClusterCriterion, HierarchyAlgorithms, LinkageMatrix, LinkageMethod};
pub use traits::kmeans::{
    KMeansAlgorithm, KMeansAlgorithms, KMeansInit, KMeansOptions, KMeansResult,
};
pub use traits::mean_shift::{MeanShiftAlgorithms, MeanShiftOptions, MeanShiftResult};
pub use traits::metrics::{ClusterMetricsAlgorithms, HCVScore};
pub use traits::mini_batch_kmeans::{MiniBatchKMeansAlgorithms, MiniBatchKMeansOptions};
pub use traits::optics::{OpticsAlgorithms, OpticsOptions, OpticsResult};
pub use traits::spectral::{
    AffinityType, LaplacianType, SpectralClusteringAlgorithms, SpectralOptions,
};
pub use validation::*;
