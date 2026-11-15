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
//! - [`roots`] - Independent scalar roots and systems of nonlinear equations
//! - [`least_squares`] - Nonlinear least squares and curve fitting
//! - [`global`] - Global optimization (escaping local minima)
//! - [`linprog`] - Linear programming (Simplex, MILP)
//! - [`conic`] - Conic programming (SOCP, SDP)
//!
//! # Example
//!
//! ```
//! # use numr::runtime::cpu::{CpuClient, CpuDevice};
//! # use solvr::optimize::scalar::ScalarOptions;
//! use solvr::optimize::OptimizationAlgorithms;
//! # let device = CpuDevice::new();
//! # let client = CpuClient::new(device.clone());
//! // Scalar root finding
//! let result = client.bisect(|x| x * x - 4.0, 0.0, 3.0, &ScalarOptions::default())?;
//! # assert!((result.root - 2.0).abs() < 1e-6);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod conic;
pub mod constrained;
pub mod error;
pub mod global;
pub mod impl_generic;
pub mod least_squares;
pub mod linprog;
pub mod minimize;
pub mod qp;
pub mod roots;
pub mod scalar;
mod traits;

// Re-export error types
pub use error::{OptimizeError, OptimizeResult};

// Re-export the main trait
pub use traits::OptimizationAlgorithms;

// Re-export result types and options
pub use minimize::{LbfgsOptions, MinimizeOptions, TensorMinimizeResult};

// Newton-CG (autograd-based second-order optimization)
pub use minimize::{NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult};

// Trust region methods (autograd-based second-order optimization)
pub use minimize::{
    TrustExactAlgorithms, TrustKrylovAlgorithms, TrustNcgAlgorithms, TrustRegionOptions,
    TrustRegionResult,
};

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

// Independent scalar roots and systems of nonlinear equations
pub use roots::{
    AndersonAlgorithms, AndersonConvergence, AndersonOptions, BatchedBisectionAlgorithms,
    BatchedBisectionOptions, BatchedBracketResult, Broyden1Algorithms,
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootQuery,
    CertifiedRootResult, CertifiedRootStatus, CertifiedRootValues, CertifiedUpperBracketAlgorithms,
    CertifiedUpperBracketOptions, CertifiedUpperBracketResult, CertifiedUpperBracketStatus,
    LevenbergMarquardtAlgorithms, NewtonSystemAlgorithms, PowellHybridAlgorithms, RootOptions,
    RootTensorResult,
};

pub use roots::{
    CertifiedMappedBisectionAlgorithms, CertifiedMappedRootProgress, CertifiedRootAdvanceAllowance,
    CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
    CertifiedRootTolerance,
};

// Least squares optimization
pub use least_squares::{LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult};

// Linear programming
pub use linprog::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
    MilpAlgorithms, MilpOptions, MilpTensorResult,
};

// Constrained optimization
pub use constrained::{
    Bounds, CobylaAlgorithms, ConstrainedOptions, ConstrainedResult, Constraint, ConstraintType,
    SlsqpAlgorithms, TrustConstrAlgorithms,
};

// Quadratic programming
pub use qp::{QpAlgorithms, QpMethod, QpOptions, QpResult};

// Conic programming
pub use conic::{
    SdpAlgorithms, SdpOptions, SdpResult, SocConstraint, SocpAlgorithms, SocpOptions, SocpResult,
};

// SHGO global optimization
pub use global::{ShgoAlgorithms, ShgoResult};
