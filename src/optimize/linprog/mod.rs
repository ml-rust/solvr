//! Linear Programming algorithms.
//!
//! This module provides linear programming solvers including:
//! - `linprog` - Simplex method for linear programming
//! - `milp` - Mixed-integer linear programming via branch-and-bound
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! linprog/
//! ├── mod.rs    # Trait definition + types (exports only)
//! ├── cpu.rs    # CPU impl + scalar convenience functions
//! ├── cuda.rs   # CUDA impl (pure delegation)
//! └── wgpu.rs   # WebGPU impl (pure delegation)
//! ```
//!
//! Generic implementations live in `optimize/impl_generic/linprog/`.

mod cpu;
mod milp;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::error::{OptimizeError, OptimizeResult};

// Re-export CPU implementation for convenience
pub use cpu::*;
pub use milp::{milp, MilpOptions, MilpResult};

/// Options for linear programming solvers.
#[derive(Debug, Clone)]
pub struct LinProgOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for optimality.
    pub tol: f64,
    /// Whether to presolve (remove redundant constraints).
    pub presolve: bool,
}

impl Default for LinProgOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-9,
            presolve: true,
        }
    }
}

/// Linear constraints for LP problems (scalar API).
#[derive(Debug, Clone, Default)]
pub struct LinearConstraints {
    /// Inequality constraint matrix (A_ub * x <= b_ub).
    pub a_ub: Option<Vec<Vec<f64>>>,
    /// Inequality constraint bounds.
    pub b_ub: Option<Vec<f64>>,
    /// Equality constraint matrix (A_eq * x == b_eq).
    pub a_eq: Option<Vec<Vec<f64>>>,
    /// Equality constraint bounds.
    pub b_eq: Option<Vec<f64>>,
    /// Variable bounds as (lower, upper) pairs.
    pub bounds: Option<Vec<(f64, f64)>>,
}

/// Result of linear programming optimization (scalar API).
#[derive(Debug, Clone)]
pub struct LinProgResult {
    /// Optimal solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value.
    pub fun: f64,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Number of iterations performed.
    pub nit: usize,
    /// Status message.
    pub message: String,
    /// Slack variables for inequality constraints.
    pub slack: Vec<f64>,
}

/// Algorithmic contract for linear programming operations.
///
/// All backends implementing linear programming MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait LinProgAlgorithms<R: Runtime> {
    /// Solve a linear programming problem using the Simplex method.
    ///
    /// Minimize: c^T * x
    /// Subject to:
    ///   A_ub * x <= b_ub
    ///   A_eq * x == b_eq
    ///   lower_bounds <= x <= upper_bounds
    fn linprog(
        &self,
        c: &Tensor<R>,
        constraints: &LinProgTensorConstraints<R>,
        options: &LinProgOptions,
    ) -> Result<LinProgTensorResult<R>>;
}

/// Tensor-based linear constraints.
#[derive(Debug, Clone)]
pub struct LinProgTensorConstraints<R: Runtime> {
    /// Inequality constraint matrix (A_ub * x <= b_ub)
    pub a_ub: Option<Tensor<R>>,
    /// Inequality constraint bounds
    pub b_ub: Option<Tensor<R>>,
    /// Equality constraint matrix (A_eq * x == b_eq)
    pub a_eq: Option<Tensor<R>>,
    /// Equality constraint bounds
    pub b_eq: Option<Tensor<R>>,
    /// Variable lower bounds
    pub lower_bounds: Option<Tensor<R>>,
    /// Variable upper bounds
    pub upper_bounds: Option<Tensor<R>>,
}

/// Result from tensor-based linear programming.
#[derive(Debug, Clone)]
pub struct LinProgTensorResult<R: Runtime> {
    /// Optimal solution vector
    pub x: Tensor<R>,
    /// Optimal objective value
    pub fun: f64,
    /// Whether optimization succeeded
    pub success: bool,
    /// Number of iterations performed
    pub nit: usize,
    /// Status message
    pub message: String,
    /// Slack variables for inequality constraints
    pub slack: Tensor<R>,
}

/// Validate constraint dimensions.
pub(crate) fn validate_constraints(n: usize, constraints: &LinearConstraints) -> OptimizeResult<()> {
    if let Some(ref a_ub) = constraints.a_ub {
        for (i, row) in a_ub.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: A_ub row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    ),
                });
            }
        }
        if let Some(ref b_ub) = constraints.b_ub {
            if b_ub.len() != a_ub.len() {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: b_ub has {} elements, A_ub has {} rows",
                        b_ub.len(),
                        a_ub.len()
                    ),
                });
            }
        } else {
            return Err(OptimizeError::InvalidInput {
                context: "linprog: A_ub provided but b_ub is missing".to_string(),
            });
        }
    }

    if let Some(ref a_eq) = constraints.a_eq {
        for (i, row) in a_eq.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: A_eq row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    ),
                });
            }
        }
        if let Some(ref b_eq) = constraints.b_eq {
            if b_eq.len() != a_eq.len() {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: b_eq has {} elements, A_eq has {} rows",
                        b_eq.len(),
                        a_eq.len()
                    ),
                });
            }
        } else {
            return Err(OptimizeError::InvalidInput {
                context: "linprog: A_eq provided but b_eq is missing".to_string(),
            });
        }
    }

    if let Some(ref bounds) = constraints.bounds {
        if bounds.len() != n {
            return Err(OptimizeError::InvalidInput {
                context: format!("linprog: bounds has {} elements, expected {}", bounds.len(), n),
            });
        }
        for (i, &(lb, ub)) in bounds.iter().enumerate() {
            if lb > ub {
                return Err(OptimizeError::InvalidInterval {
                    a: lb,
                    b: ub,
                    context: format!("linprog: invalid bounds for variable {}", i),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = LinProgOptions::default();
        assert_eq!(opts.max_iter, 1000);
        assert!((opts.tol - 1e-9).abs() < 1e-12);
    }
}
