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
//! ## Basic Methods (Tensor-based)
//!
//! - [`IntegrationAlgorithms::trapezoid`] - Trapezoidal rule with tensor support
//! - [`IntegrationAlgorithms::simpson`] - Simpson's rule with tensor support
//! - [`IntegrationAlgorithms::fixed_quad`] - Gaussian quadrature with tensor functions
//!
//! ## Legacy Scalar Methods
//!
//! - [`trapezoid`] - Trapezoidal rule for slice data
//! - [`cumulative_trapezoid`] - Cumulative trapezoidal integration
//! - [`simpson`] - Simpson's rule
//! - [`fixed_quad`] - Fixed-order Gaussian quadrature
//! - [`quad`] - Adaptive Gauss-Kronrod quadrature
//! - [`romberg`] - Romberg integration
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
pub mod quadrature;
#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export error types
pub use error::{IntegrateError, IntegrateResult};

// Re-export quadrature functions (legacy scalar API)
pub use quadrature::{
    GaussLegendreQuadrature,
    QuadOptions,
    QuadResult,
    RombergOptions,
    // Basic quadrature
    cumulative_trapezoid,
    // Gaussian quadrature
    fixed_quad,
    // Adaptive quadrature
    quad,
    romberg,
    simpson,
    trapezoid,
};

// Re-export ODE types and functions
pub use ode::{ODEMethod, ODEOptions, ODEResult, ODESolution, StepSizeController, solve_ivp};

/// Trait for integration algorithms that work across all Runtime backends.
///
/// This trait provides a unified interface for:
/// - Trapezoidal integration
/// - Simpson's rule
/// - Gaussian quadrature
/// - Cumulative integration
///
/// All methods work with `Tensor<R>` for GPU acceleration and batch operations.
///
/// # Example
///
/// ```ignore
/// use solvr::integrate::IntegrationAlgorithms;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
///
/// // Integrate y = x^2 from 0 to 1
/// let x = Tensor::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], &device);
/// let y = Tensor::from_slice(&[0.0, 0.0625, 0.25, 0.5625, 1.0], &[5], &device);
/// let result = client.trapezoid(&y, &x)?;
/// ```
pub trait IntegrationAlgorithms<R: Runtime> {
    /// Trapezoidal rule integration.
    ///
    /// Computes ∫y dx using the composite trapezoidal rule.
    ///
    /// # Arguments
    /// * `y` - Function values (1D or 2D for batch)
    /// * `x` - Sample points (1D)
    ///
    /// # Returns
    /// * 0-D tensor for 1D input
    /// * 1-D tensor for 2D input (one value per row)
    fn trapezoid(&self, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Trapezoidal rule with uniform spacing.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `dx` - Uniform spacing between points
    fn trapezoid_uniform(&self, y: &Tensor<R>, dx: f64) -> Result<Tensor<R>>;

    /// Cumulative trapezoidal integration.
    ///
    /// Returns running integral values.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `x` - Sample points (optional, uses dx if None)
    /// * `dx` - Uniform spacing (used if x is None)
    fn cumulative_trapezoid(
        &self,
        y: &Tensor<R>,
        x: Option<&Tensor<R>>,
        dx: f64,
    ) -> Result<Tensor<R>>;

    /// Simpson's rule integration.
    ///
    /// Uses Simpson's 1/3 rule for higher accuracy than trapezoidal.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `x` - Sample points (optional, uses dx if None)
    /// * `dx` - Uniform spacing (used if x is None)
    fn simpson(&self, y: &Tensor<R>, x: Option<&Tensor<R>>, dx: f64) -> Result<Tensor<R>>;

    /// Fixed-order Gaussian quadrature.
    ///
    /// Integrates a tensor-valued function from a to b using
    /// n-point Gauss-Legendre quadrature.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `n` - Number of quadrature points
    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
