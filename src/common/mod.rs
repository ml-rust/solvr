//! Common utilities shared across solvr modules.
//!
//! This module contains shared infrastructure used by multiple solvr submodules
//! (optimize, integrate, interpolate, etc.).

pub mod arithmetic;
pub mod jacobian;

pub use jacobian::{jacobian_autograd, jvp_autograd};
