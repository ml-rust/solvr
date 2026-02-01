//! Optimization algorithms for solvr.
//!
//! This module provides Runtime-first optimization algorithms that work across
//! CPU, CUDA, and WebGPU backends.
//!
//! # Architecture
//!
//! All algorithms implement the [`OptimizationAlgorithms`] trait and use
//! `Tensor<R>` for multivariate methods, enabling GPU acceleration.
//!
//! # Modules
//!
//! - [`scalar`] - Univariate (1D) root finding and minimization
//! - [`minimize`] - Multivariate unconstrained minimization
//! - [`roots`] - Multivariate root finding (systems of nonlinear equations)
//! - [`least_squares`] - Nonlinear least squares and curve fitting
//! - [`global`] - Global optimization (escaping local minima)
//! - [`linprog`] - Linear programming (Simplex, MILP)
//!
//! # Example
//!
//! ```ignore
//! use solvr::optimize::OptimizationAlgorithms;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Scalar root finding
//! let result = client.bisect(|x| x * x - 4.0, 0.0, 3.0, &ScalarOptions::default())?;
//! assert!((result.root - 2.0).abs() < 1e-6);
//!
//! // Multivariate minimization with tensors
//! let x0 = Tensor::from_slice(&[1.0, 1.0], &[2], &device);
//! let result = client.bfgs(|x| Ok(x.to_vec().iter().map(|xi| xi * xi).sum()), &x0, &opts)?;
//! ```

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod error;
pub mod global;
pub mod impl_generic;
pub mod least_squares;
pub mod linprog;
pub mod minimize;
pub mod roots;
pub mod scalar;
pub(crate) mod utils;
#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use scalar::{MinimizeResult, RootResult, ScalarOptions};

// Re-export error types
pub use error::{OptimizeError, OptimizeResult};

// Re-export result types and options
pub use impl_generic::TensorMinimizeResult;
pub use minimize::MinimizeOptions;

// Re-export scalar optimization (1D) - these are inherently scalar, not tensor
pub use scalar::{
    MinimizeResult as ScalarMinResult, RootResult as ScalarRootResult,
    ScalarOptions as ScalarOpts, bisect, brentq, minimize_scalar_bounded, minimize_scalar_brent,
    minimize_scalar_golden, newton, ridder, secant,
};

// TODO: These modules still use scalar &[f64] APIs and need tensor migration:
// - global: differential_evolution, simulated_annealing, dual_annealing, basinhopping
// - roots: newton_system, broyden1, levenberg_marquardt
// - least_squares: leastsq, least_squares, curve_fit
// - linprog: linprog, milp
//
// For now, re-export them but they should be rewritten with tensor operations.
pub use global::{
    GlobalOptions, GlobalResult, basinhopping, differential_evolution, dual_annealing,
    simulated_annealing,
};
pub use least_squares::{
    LeastSquaresOptions, LeastSquaresResult, curve_fit, least_squares, leastsq,
};
pub use linprog::{
    LinProgOptions, LinProgResult, LinearConstraints, MilpOptions, MilpResult, linprog, milp,
};
pub use roots::{MultiRootResult, RootOptions, broyden1, levenberg_marquardt, newton_system};

/// Trait for optimization algorithms that work across all Runtime backends.
///
/// This trait provides a unified interface for:
/// - Scalar root finding (bisect, brentq, newton)
/// - Scalar minimization (brent)
/// - Multivariate minimization (BFGS, Nelder-Mead, Powell, CG)
///
/// # Example
///
/// ```ignore
/// use solvr::optimize::OptimizationAlgorithms;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
///
/// // Find root of x^2 - 4 = 0
/// let result = client.bisect(|x| x * x - 4.0, 0.0, 3.0, &ScalarOptions::default())?;
/// ```
pub trait OptimizationAlgorithms<R: Runtime> {
    /// Bisection method for scalar root finding.
    ///
    /// Finds a root of `f` in the interval [a, b].
    fn bisect<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &ScalarOptions,
    ) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64;

    /// Brent's method for scalar root finding.
    ///
    /// Combines bisection, secant, and inverse quadratic interpolation.
    fn brentq<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &ScalarOptions,
    ) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64;

    /// Newton's method for scalar root finding.
    ///
    /// Requires the derivative `df` of the function.
    fn newton<F, DF>(
        &self,
        f: F,
        df: DF,
        x0: f64,
        options: &ScalarOptions,
    ) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64;

    /// Brent's method for scalar minimization.
    fn minimize_scalar_brent<F>(
        &self,
        f: F,
        bracket: Option<(f64, f64, f64)>,
        options: &ScalarOptions,
    ) -> OptimizeResult<MinimizeResult>
    where
        F: Fn(f64) -> f64;

    /// BFGS quasi-Newton method for multivariate minimization.
    ///
    /// Uses tensor-based computation for GPU acceleration.
    fn bfgs<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Nelder-Mead simplex method for multivariate minimization.
    ///
    /// Derivative-free method suitable for non-smooth functions.
    fn nelder_mead<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Powell's method for multivariate minimization.
    ///
    /// Derivative-free direction set method.
    fn powell<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Conjugate gradient method for multivariate minimization.
    ///
    /// Polak-Ribière variant with automatic restarts.
    fn conjugate_gradient<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
