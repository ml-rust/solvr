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

pub mod error;
pub mod global;
pub mod impl_generic;
pub mod least_squares;
pub mod linprog;
pub mod minimize;
pub mod roots;
pub mod scalar;
mod traits;

// Re-export error types
pub use error::{OptimizeError, OptimizeResult};

// Re-export the main trait
pub use traits::OptimizationAlgorithms;

// Re-export result types and options
pub use minimize::{LbfgsOptions, MinimizeOptions, TensorMinimizeResult};

// Re-export scalar optimization (1D) - these are inherently scalar, not tensor
pub use scalar::{
    MinimizeResult as ScalarMinResult, RootResult as ScalarRootResult, ScalarOptions as ScalarOpts,
    bisect, brentq, minimize_scalar_bounded, minimize_scalar_brent, minimize_scalar_golden, newton,
    ridder, secant,
};

// Global optimization
pub use global::{
    BasinHoppingAlgorithms, BasinHoppingResult, DifferentialEvolutionAlgorithms,
    DifferentialEvolutionResult, DualAnnealingAlgorithms, DualAnnealingResult, GlobalOptions,
    SimulatedAnnealingAlgorithms, SimulatedAnnealingResult,
};

// Root finding (systems of nonlinear equations)
pub use roots::{
    Broyden1Algorithms, LevenbergMarquardtAlgorithms, NewtonSystemAlgorithms, RootOptions,
    RootTensorResult,
};

// Least squares optimization
pub use least_squares::{LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult};

// Linear programming
pub use linprog::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
    MilpAlgorithms, MilpOptions, MilpTensorResult,
};
