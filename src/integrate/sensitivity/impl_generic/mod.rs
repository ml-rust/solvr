//! Generic implementations for sensitivity analysis.

pub mod adjoint_ode;
pub mod checkpointing;

pub use adjoint_ode::adjoint_sensitivity_impl;
pub use checkpointing::CheckpointManager;
