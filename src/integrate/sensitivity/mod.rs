//! Adjoint sensitivity analysis for ODE solvers.
//!
//! This module provides memory-efficient computation of parameter gradients
//! using the adjoint (continuous) sensitivity method.
//!
//! # Overview
//!
//! For an ODE dy/dt = f(t, y, p) with cost function J = g(y(T)), we want to
//! compute ∂J/∂p. The adjoint method achieves this with O(n_params + n_states)
//! cost regardless of the number of time steps.
//!
//! # Memory Efficiency
//!
//! Instead of storing the full forward trajectory (O(n_steps) memory), we use
//! checkpointing to achieve O(n_checkpoints) memory with some recomputation
//! during the backward pass.
//!
//! # Example
//!
//! ```no_run
//! use solvr::integrate::sensitivity::{SensitivityOptions, AdjointSensitivityAlgorithms};
//! use numr::autograd::Var;
//! use numr::runtime::cpu::{CpuClient, CpuRuntime};
//!
//! // ODE: dy/dt = -k*y, y(0) = 1
//! // Cost: J = y(T)²
//!
//! let f = |_t: &Var<CpuRuntime>, _y: &Var<CpuRuntime>, _p: &Var<CpuRuntime>, _c: &CpuClient| {
//!     unimplemented!()
//! };
//!
//! let g = |_y: &Var<CpuRuntime>, _c: &CpuClient| {
//!     unimplemented!()
//! };
//! ```

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
pub mod wgpu;

// Re-exports
pub use impl_generic::adjoint_ode::adjoint_sensitivity_impl;
pub use impl_generic::checkpointing::CheckpointManager;
pub use traits::{
    AdjointSensitivityAlgorithms, Checkpoint, CheckpointStrategy, SensitivityOptions,
    SensitivityResult,
};
