//! Shared types for constrained optimization.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub type ConstraintFn<'a, R> = dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>> + 'a;

/// Type of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Equality constraint: fun(x) = 0
    Equality,
    /// Inequality constraint: fun(x) >= 0
    Inequality,
}

/// A nonlinear constraint for constrained optimization.
///
/// For equality constraints: fun(x) = 0
/// For inequality constraints: fun(x) >= 0
pub struct Constraint<'a, R: Runtime> {
    /// Type of constraint (equality or inequality).
    pub kind: ConstraintType,
    /// Constraint function. Returns a tensor of constraint values.
    pub fun: &'a ConstraintFn<'a, R>,
    /// Optional Jacobian of the constraint function.
    /// If None, finite differences will be used.
    pub jac: Option<&'a ConstraintFn<'a, R>>,
}

/// Variable bounds for constrained optimization.
#[derive(Debug, Clone)]
pub struct Bounds<R: Runtime> {
    /// Lower bounds (None means -infinity for all variables).
    pub lower: Option<Tensor<R>>,
    /// Upper bounds (None means +infinity for all variables).
    pub upper: Option<Tensor<R>>,
}

impl<R: Runtime> Default for Bounds<R> {
    fn default() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }
}

/// Options for constrained optimization algorithms.
#[derive(Debug, Clone)]
pub struct ConstrainedOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for optimality (KKT conditions).
    pub tol: f64,
    /// Step size for finite difference approximation.
    pub eps: f64,
    /// Tolerance for constraint violation.
    pub constraint_tol: f64,
}

impl Default for ConstrainedOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            eps: 1e-8,
            constraint_tol: 1e-8,
        }
    }
}

/// Result of constrained optimization.
#[derive(Debug, Clone)]
pub struct ConstrainedResult<R: Runtime> {
    /// Solution vector.
    pub x: Tensor<R>,
    /// Objective function value at solution.
    pub fun: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Maximum constraint violation at solution.
    pub constraint_violation: f64,
    /// Status message.
    pub message: String,
}
