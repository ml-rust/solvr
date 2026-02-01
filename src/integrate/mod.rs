//! Numerical integration and ODE solvers for solvr.
//!
//! This module provides Runtime-first numerical methods for:
//! - **Quadrature**: Numerical integration of functions (definite integrals)
//! - **ODE Solvers**: Initial value problem solvers for ordinary differential equations
//!
//! # Architecture
//!
//! All algorithms implement the [`IntegrationAlgorithms`] trait and use
//! `Tensor<R>` for data, enabling GPU acceleration and batch operations.
//!
//! # Quadrature Methods
//!
//! - [`IntegrationAlgorithms::trapezoid`] - Trapezoidal rule with tensor support
//! - [`IntegrationAlgorithms::simpson`] - Simpson's rule with tensor support
//! - [`IntegrationAlgorithms::fixed_quad`] - Gaussian quadrature with tensor functions
//! - [`IntegrationAlgorithms::quad`] - Adaptive Gauss-Kronrod quadrature
//! - [`IntegrationAlgorithms::romberg`] - Romberg integration via Richardson extrapolation
//!
//! # ODE Solvers
//!
//! ## Unified Interface
//!
//! - [`solve_ivp`] - Main entry point for solving initial value problems
//!
//! ## Available Methods
//!
//! - **RK23**: Bogacki-Shampine 2(3) - Low accuracy, fast
//! - **RK45**: Dormand-Prince 4(5) - General purpose (default)
//! - **DOP853**: Dormand-Prince 8(5,3) - High accuracy
//!
//! # Example
//!
//! ```ignore
//! use solvr::integrate::IntegrationAlgorithms;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Tensor-based integration
//! let x = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
//! let y = Tensor::from_slice(&[0.0, 0.25, 1.0], &[3], &device);  // y = x^2
//! let result = client.trapezoid(&y, &x)?;
//! ```

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod error;
pub mod impl_generic;
pub mod ode;
mod traits;
#[cfg(feature = "wgpu")]
mod wgpu;

// Re-export error types
pub use error::{IntegrateError, IntegrateResult};

// Re-export ODE types
pub use ode::{ODEMethod, ODEOptions};

// Re-export tensor-based ODE types and functions
pub use impl_generic::ode::{ODEResultTensor, solve_ivp_impl};

// Re-export the main trait and types
pub use traits::{IntegrationAlgorithms, QuadOptions, QuadResult, RombergOptions};
