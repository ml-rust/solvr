use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::integrate::error::IntegrateResult;
use crate::integrate::{ODEOptions, ODEResultTensor};

use super::{QuadOptions, QuadResult, RombergOptions};

/// Trait for integration algorithms that work across all Runtime backends.
///
/// This trait provides a unified interface for:
/// - Trapezoidal integration
/// - Simpson's rule
/// - Gaussian quadrature
/// - Adaptive quadrature
/// - Romberg integration
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

    /// Adaptive Gauss-Kronrod quadrature.
    ///
    /// Uses the G7-K15 rule (7-point Gauss, 15-point Kronrod) with adaptive
    /// interval subdivision to achieve the requested tolerance.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points and returns tensor of values
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `options` - Quadrature options (tolerances, max subdivisions)
    ///
    /// # Returns
    /// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
    fn quad<F>(&self, f: F, a: f64, b: f64, options: &QuadOptions) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Romberg integration using Richardson extrapolation.
    ///
    /// Applies Richardson extrapolation to the trapezoidal rule to achieve
    /// high accuracy for smooth functions.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points and returns tensor of values
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `options` - Integration options (tolerances, max levels)
    ///
    /// # Returns
    /// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
    fn romberg<F>(&self, f: F, a: f64, b: f64, options: &RombergOptions) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Solve an initial value problem for ODEs using tensor operations.
    ///
    /// Solves the system dy/dt = f(t, y) with initial condition y(t0) = y0.
    /// All computation stays on device - no GPU→CPU→GPU roundtrips.
    /// Time is passed as a scalar tensor (shape [1]) to enable device-resident computation.
    ///
    /// # Arguments
    /// * `f` - Right-hand side function f(t, y) -> dy/dt, where t is a scalar tensor [1]
    /// * `t_span` - Integration interval [t0, tf]
    /// * `y0` - Initial condition as a 1-D tensor
    /// * `options` - Solver options (method, tolerances, step bounds)
    ///
    /// # Returns
    /// An [`ODEResultTensor`] with solution trajectory stored as tensors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use solvr::integrate::{IntegrationAlgorithms, ODEOptions};
    /// use numr::runtime::cpu::{CpuClient, CpuDevice};
    ///
    /// let device = CpuDevice::new();
    /// let client = CpuClient::new(device.clone());
    ///
    /// // Solve dy/dt = -y, y(0) = 1
    /// // Note: t is a tensor [1], y is a tensor [n]
    /// let y0 = Tensor::from_slice(&[1.0], &[1], &device);
    /// let result = client.solve_ivp(
    ///     |_t, y| client.mul_scalar(y, -1.0),
    ///     [0.0, 5.0],
    ///     &y0,
    ///     &ODEOptions::default(),
    /// )?;
    /// ```
    fn solve_ivp<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        options: &ODEOptions,
    ) -> IntegrateResult<ODEResultTensor<R>>
    where
        F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>;
}
