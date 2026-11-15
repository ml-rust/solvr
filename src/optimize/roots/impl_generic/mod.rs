//! Tensor-based implementations of multivariate root finding algorithms.

mod anderson;
mod broyden;
pub mod helpers;
mod levenberg_marquardt;
mod newton;
mod powell_hybrid;

pub use anderson::{anderson_impl, anderson_with_convergence_impl};
pub use broyden::broyden1_impl;
pub use helpers::{jacobian_forward_impl, jvp_impl};
pub use levenberg_marquardt::levenberg_marquardt_impl;
pub use newton::newton_system_impl;
pub use powell_hybrid::powell_hybrid_impl;

pub use super::traits::root_result::TensorRootResult;

pub mod bisection;
pub use bisection::bisect_batched_impl;

pub mod batched_newton;
pub use batched_newton::newton_batched_impl;

pub(crate) mod certified_bisection;
pub use certified_bisection::bisect_certified_impl;
mod certified_mapped_bisection;
pub use certified_mapped_bisection::{
    bisect_certified_mapped_impl, refresh_certified_mapped_selected_impl,
    resume_certified_mapped_impl, resume_certified_mapped_selected_impl,
};
pub(crate) mod certified;
mod certified_upper_bracket;
pub use certified_upper_bracket::bracket_upper_certified_impl;
