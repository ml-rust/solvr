//! Multivariate unconstrained minimization.
//!
//! This module provides algorithms and configuration types for minimization.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod impl_generic;
mod traits;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::{MinimizeOptions, TensorMinimizeResult};
