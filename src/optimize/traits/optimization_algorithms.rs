//! Unified optimization algorithms trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::LbfgsOptions;
use crate::optimize::minimize::{MinimizeOptions, TensorMinimizeResult};
use crate::optimize::scalar::{MinimizeResult, RootResult, ScalarOptions};

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
    /// Memory: O(n²) for storing inverse Hessian approximation.
    fn bfgs<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// L-BFGS (Limited-memory BFGS) for large-scale minimization.
    ///
    /// Memory-efficient variant of BFGS using O(mn) memory instead of O(n²).
    /// Stores m recent correction pairs instead of full inverse Hessian.
    /// Ideal for problems with thousands to millions of parameters.
    fn lbfgs<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &LbfgsOptions,
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
    /// Polak-Ribiere variant with automatic restarts.
    fn conjugate_gradient<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
