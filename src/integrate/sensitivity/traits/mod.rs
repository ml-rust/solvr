//! Trait definitions for sensitivity analysis.

mod adjoint;
mod types;

pub use adjoint::AdjointSensitivityAlgorithms;
pub use types::{Checkpoint, CheckpointStrategy, SensitivityOptions, SensitivityResult};
