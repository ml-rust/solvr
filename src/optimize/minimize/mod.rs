//! Multivariate unconstrained minimization.
//!
//! This module provides algorithms and configuration types for minimization.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use impl_generic::LbfgsOptions;
pub use traits::{
    MinimizeOptions, NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult, TensorMinimizeResult,
};
