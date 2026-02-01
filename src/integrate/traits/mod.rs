//! Trait definitions and types for integration algorithms.

mod algorithms;
mod types;

pub use algorithms::IntegrationAlgorithms;
pub use types::{QuadOptions, QuadResult, RombergOptions};
