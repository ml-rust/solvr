//! Generic implementations of multivariate minimization algorithms.
//!
//! These implementations work across all Runtime backends using tensor operations.

pub mod bfgs;
pub mod conjugate_gradient;
pub mod helpers;
pub mod lbfgs;
pub mod nelder_mead;
pub mod powell;
pub mod utils;

// Re-export main types and functions
pub use bfgs::bfgs_impl;
pub use conjugate_gradient::conjugate_gradient_impl;
pub use helpers::TensorMinimizeResult;
pub use lbfgs::{LbfgsOptions, lbfgs_impl};
pub use nelder_mead::nelder_mead_impl;
pub use powell::powell_impl;
