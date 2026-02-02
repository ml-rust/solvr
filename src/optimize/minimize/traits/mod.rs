//! Trait definitions and types for multivariate minimization.

pub mod newton_cg;
mod types;

pub use newton_cg::{NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult};
pub use types::{MinimizeOptions, TensorMinimizeResult};
