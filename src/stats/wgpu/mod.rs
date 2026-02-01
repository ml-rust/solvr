//! WebGPU implementation of statistical algorithms.
//!
//! This module implements the statistical traits for WebGPU using numr's tensor operations
//! via the generic implementations.

mod descriptive;
mod hypothesis;
mod regression;

pub use descriptive::*;
pub use hypothesis::*;
pub use regression::*;
