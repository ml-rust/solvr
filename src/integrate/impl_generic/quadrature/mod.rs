//! Tensor-based quadrature implementations.
//!
//! This module provides numerical integration algorithms using tensor operations.
//! All implementations are generic over `R: Runtime<DType = DType>` for multi-backend support.

mod certified_simpson;
mod fixed_quad;
mod monte_carlo;
mod nquad;
mod qmc;
pub mod quad;
mod romberg;
mod simpson;
mod tanh_sinh;
mod trapezoid;

pub use certified_simpson::certified_simpson_impl;
pub use fixed_quad::fixed_quad_impl;
pub use monte_carlo::monte_carlo_impl;
pub use nquad::{dblquad_impl, nquad_impl, tplquad_impl};
pub use qmc::qmc_impl;
pub use quad::quad_impl;
pub use romberg::romberg_impl;
pub use simpson::simpson_impl;
pub use tanh_sinh::tanh_sinh_impl;
pub use trapezoid::{cumulative_trapezoid_impl, trapezoid_impl, trapezoid_uniform_impl};
