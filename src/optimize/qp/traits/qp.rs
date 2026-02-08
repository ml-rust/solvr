//! Quadratic programming algorithm traits and types.
//!
//! Solves: min 0.5*x'*Q*x + c'*x
//!         s.t. A_eq*x = b_eq
//!              A_ineq*x >= b_ineq
//!              lower <= x <= upper

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;

/// QP solver method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpMethod {
    /// Active set method (efficient for small-medium problems).
    ActiveSet,
    /// Interior point method (efficient for larger problems).
    #[default]
    InteriorPoint,
}

/// Options for QP solver.
#[derive(Debug, Clone)]
pub struct QpOptions {
    /// Solver method.
    pub method: QpMethod,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for convergence.
    pub tol: f64,
}

impl Default for QpOptions {
    fn default() -> Self {
        Self {
            method: QpMethod::InteriorPoint,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

/// Result of QP solver.
#[derive(Debug, Clone)]
pub struct QpResult<R: Runtime> {
    /// Solution vector.
    pub x: Tensor<R>,
    /// Optimal objective value.
    pub fun: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Dual variables for equality constraints.
    pub dual_eq: Option<Tensor<R>>,
    /// Dual variables for inequality constraints.
    pub dual_ineq: Option<Tensor<R>>,
}

/// Trait for quadratic programming.
pub trait QpAlgorithms<R: Runtime> {
    /// Solve a quadratic program.
    ///
    /// min  0.5*x'*Q*x + c'*x
    /// s.t. A_eq*x = b_eq       (if provided)
    ///      A_ineq*x >= b_ineq   (if provided)
    ///      lower <= x <= upper   (if provided)
    ///
    /// # Arguments
    ///
    /// * `q` - Symmetric positive (semi-)definite matrix [n, n]
    /// * `c` - Linear cost vector [n]
    /// * `a_eq` - Equality constraint matrix [m_eq, n] (optional)
    /// * `b_eq` - Equality constraint RHS [m_eq] (optional)
    /// * `a_ineq` - Inequality constraint matrix [m_ineq, n] (optional)
    /// * `b_ineq` - Inequality constraint RHS [m_ineq] (optional)
    /// * `options` - Solver options
    #[allow(clippy::too_many_arguments)]
    fn solve_qp(
        &self,
        q: &Tensor<R>,
        c: &Tensor<R>,
        a_eq: Option<&Tensor<R>>,
        b_eq: Option<&Tensor<R>>,
        a_ineq: Option<&Tensor<R>>,
        b_ineq: Option<&Tensor<R>>,
        options: &QpOptions,
    ) -> OptimizeResult<QpResult<R>>;
}
