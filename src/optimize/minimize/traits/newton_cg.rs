//! Newton-CG (Truncated Newton) algorithm trait and types.
//!
//! Newton-CG combines Newton's method with conjugate gradient for solving
//! the Newton system H·p = -∇f. Instead of forming and inverting the Hessian,
//! it uses Hessian-vector products (HVP) within a CG inner loop.
//!
//! # When to Use Newton-CG
//!
//! - **Large-scale problems**: n > 1000 where forming H is prohibitive
//! - **Smooth, twice-differentiable functions**: Newton's method needs Hessian
//! - **Near a local minimum**: Newton converges quadratically near minima
//! - **Well-conditioned Hessian**: CG converges slowly for ill-conditioned problems
//!
//! # Memory Usage
//!
//! - BFGS: O(n²) for inverse Hessian approximation
//! - L-BFGS: O(mn) for m correction pairs
//! - Newton-CG: O(n) per iteration (no Hessian storage)

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::TensorMinimizeResult;

/// Options for Newton-CG optimization.
#[derive(Debug, Clone)]
pub struct NewtonCGOptions {
    /// Maximum number of outer (Newton) iterations
    pub max_iter: usize,
    /// Maximum number of inner (CG) iterations per Newton step
    /// If None, defaults to min(n, 200)
    pub max_cg_iter: Option<usize>,
    /// Tolerance for gradient norm convergence
    pub g_tol: f64,
    /// Tolerance for function value change
    pub f_tol: f64,
    /// Tolerance for argument change
    pub x_tol: f64,
    /// Relative tolerance for CG convergence (||r|| < cg_tol * ||g||)
    pub cg_tol: f64,
    /// Initial trust region radius (None for line search instead)
    pub trust_radius: Option<f64>,
}

impl Default for NewtonCGOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            max_cg_iter: None,
            g_tol: 1e-8,
            f_tol: 1e-8,
            x_tol: 1e-8,
            cg_tol: 0.1, // Inexact Newton: don't solve CG too precisely
            trust_radius: None,
        }
    }
}

/// Result type for Newton-CG optimization (extends base result).
#[derive(Debug, Clone)]
pub struct NewtonCGResult<R: Runtime> {
    /// Solution vector
    pub x: Tensor<R>,
    /// Function value at solution
    pub fun: f64,
    /// Number of outer (Newton) iterations
    pub iterations: usize,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Total number of gradient evaluations
    pub ngrad: usize,
    /// Total number of Hessian-vector products
    pub nhvp: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final gradient norm
    pub grad_norm: f64,
}

impl<R: Runtime> From<NewtonCGResult<R>> for TensorMinimizeResult<R> {
    fn from(result: NewtonCGResult<R>) -> Self {
        TensorMinimizeResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev + result.ngrad + result.nhvp, // Total evaluations
            converged: result.converged,
        }
    }
}

/// Trait for Newton-CG optimization algorithm.
///
/// Newton-CG uses autograd for exact gradients and Hessian-vector products,
/// enabling quadratic convergence without O(n²) memory.
pub trait NewtonCGAlgorithms<R: Runtime> {
    /// Newton-CG (Truncated Newton) optimization with autograd.
    ///
    /// Minimizes a scalar function f: ℝⁿ → ℝ using Newton's method with
    /// conjugate gradient for solving the Newton system. Uses automatic
    /// differentiation for exact gradients and Hessian-vector products.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a `Var<R>` and returns a scalar `Var<R>`.
    ///   Must be differentiable (use autograd operations like `var_mul`, etc.)
    /// * `x0` - Initial guess as a Tensor
    /// * `options` - Algorithm options
    ///
    /// # Returns
    ///
    /// `NewtonCGResult` containing the solution and convergence information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use numr::autograd::{Var, var_mul, var_sum};
    ///
    /// // Minimize f(x) = sum(x²)
    /// let result = client.newton_cg(
    ///     |x, c| {
    ///         let x_sq = var_mul(x, x, c)?;
    ///         var_sum(&x_sq, &[], false, c)
    ///     },
    ///     &x0,
    ///     &NewtonCGOptions::default(),
    /// )?;
    /// ```
    fn newton_cg<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &NewtonCGOptions,
    ) -> OptimizeResult<NewtonCGResult<R>>
    where
        F: Fn(&Var<R>, &Self) -> NumrResult<Var<R>>;
}
