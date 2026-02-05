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
//! ```ignore
//! use solvr::integrate::sensitivity::{SensitivityOptions, AdjointSensitivityAlgorithms};
//! use numr::autograd::{Var, var_mul, var_mul_scalar};
//!
//! // ODE: dy/dt = -k*y, y(0) = 1
//! // Cost: J = y(T)²
//! // Gradient: ∂J/∂k = -2*T*y(T)²
//!
//! let f = |t: &Var<R>, y: &Var<R>, p: &Var<R>, c: &C| {
//!     // -k * y
//!     let ky = var_mul(p, y, c)?;
//!     var_mul_scalar(&ky, -1.0, c)
//! };
//!
//! let g = |y: &Var<R>, c: &C| {
//!     // y²
//!     var_mul(y, y, c)
//! };
//!
//! let result = client.adjoint_sensitivity(
//!     f, g, [0.0, T], &y0, &p, &ode_opts, &sens_opts
//! )?;
//!
//! // result.gradient contains ∂J/∂p
//! ```

pub mod cpu;
pub mod impl_generic;
pub mod traits;

// Re-exports
pub use impl_generic::adjoint_ode::adjoint_sensitivity_impl;
pub use impl_generic::checkpointing::CheckpointManager;
pub use traits::{
    AdjointSensitivityAlgorithms, Checkpoint, CheckpointStrategy, SensitivityOptions,
    SensitivityResult,
};
