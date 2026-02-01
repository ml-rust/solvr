//! Tensor-based linear programming implementations.
//!
//! Provides simplex method for linear programming using tensor operations.

mod simplex;

pub use simplex::simplex_impl;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from tensor-based linear programming.
#[derive(Debug, Clone)]
pub struct TensorLinProgResult<R: Runtime> {
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

/// Tensor-based linear constraints.
#[derive(Debug, Clone)]
pub struct TensorLinearConstraints<R: Runtime> {
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
