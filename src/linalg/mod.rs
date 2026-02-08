//! Linear algebra algorithms.
//!
//! Matrix equation solvers for control theory and systems engineering:
//! - Sylvester equation: AX + XB = C
//! - Continuous Lyapunov: AX + XA^T = Q
//! - Discrete Lyapunov: AXA^T - X + Q = 0
//! - Continuous algebraic Riccati (CARE)
//! - Discrete algebraic Riccati (DARE)

mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::matrix_equations::MatrixEquationAlgorithms;
