//! Global optimization algorithms.
//!
//! Provides methods for finding global minima of functions,
//! avoiding local minima traps that affect local optimization methods.
//!
//! All algorithms use tensor operations and are generic over `R: Runtime`.

pub mod cpu;
pub mod impl_generic;
mod traits;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use traits::*;
