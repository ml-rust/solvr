//! Generic implementations of multivariate minimization algorithms.
//!
//! These implementations work across all Runtime backends using tensor operations.

pub mod bfgs;
pub mod conjugate_gradient;
pub mod helpers;
pub mod lbfgs;
pub mod nelder_mead;
pub mod newton_cg;
pub mod powell;
pub mod utils;

// Re-export main types and functions
pub use bfgs::bfgs_impl;
pub use conjugate_gradient::conjugate_gradient_impl;
pub use helpers::{TensorMinimizeResult, gradient_from_fn, hvp_from_fn, hvp_reverse_over_reverse};
pub use lbfgs::{LbfgsOptions, lbfgs_impl};
pub use nelder_mead::nelder_mead_impl;
pub use newton_cg::newton_cg_impl;
pub use powell::powell_impl;
