use numr::autograd::DualTensor;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::integrate::error::IntegrateResult;
use crate::integrate::ode::{
    BDFOptions, BVPOptions, LSODAOptions, RadauOptions, SymplecticOptions,
};
use crate::integrate::{ODEOptions, ODEResultTensor};

use super::{
    BVPResult, MonteCarloOptions, MonteCarloResult, NQuadOptions, QMCOptions, QuadOptions,
    QuadResult, RombergOptions, SymplecticResult, TanhSinhOptions,
};

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

    // ========================================================================
    // Advanced Quadrature Methods
    // ========================================================================

    /// Tanh-sinh (double exponential) quadrature.
    ///
    /// Highly effective for integrals with endpoint singularities or infinite
    /// derivatives at boundaries. Uses the transformation:
    /// x = tanh(π/2 * sinh(t))
    ///
    /// # Arguments
    /// * `f` - Integrand function taking tensor of evaluation points
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `options` - Integration options
    ///
    /// # Returns
    /// A [`QuadResult`] with integral value and error estimate.
    fn tanh_sinh<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &TanhSinhOptions,
    ) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Monte Carlo integration for multi-dimensional integrals.
    ///
    /// Suitable for high-dimensional integrals where deterministic methods
    /// suffer from the curse of dimensionality. Error decreases as O(1/√n).
    ///
    /// # Arguments
    /// * `f` - Integrand function taking tensor of shape [n_samples, n_dims]
    /// * `bounds` - Integration bounds for each dimension [(a1, b1), (a2, b2), ...]
    /// * `options` - Monte Carlo options (samples, method, seed)
    ///
    /// # Returns
    /// A [`MonteCarloResult`] with integral estimate and standard error.
    fn monte_carlo<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &MonteCarloOptions,
    ) -> Result<MonteCarloResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Quasi-Monte Carlo integration using low-discrepancy sequences.
    ///
    /// Similar to Monte Carlo but uses deterministic low-discrepancy sequences
    /// (Sobol, Halton) instead of random points. Faster convergence: O(1/n)
    /// vs O(1/√n) for random sampling.
    ///
    /// # Arguments
    /// * `f` - Integrand function taking tensor of shape [n_samples, n_dims]
    /// * `bounds` - Integration bounds for each dimension
    /// * `options` - QMC options (samples, sequence type)
    ///
    /// # Returns
    /// A [`QuadResult`] with integral estimate.
    fn qmc_quad<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &QMCOptions,
    ) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Double integral over a rectangular region.
    ///
    /// Computes ∫∫ f(x, y) dx dy over [a, b] × [gfun(x), hfun(x)].
    ///
    /// # Arguments
    /// * `f` - Integrand function f(x, y) where x, y are tensors
    /// * `a`, `b` - Outer integration bounds (x direction)
    /// * `gfun`, `hfun` - Functions defining inner bounds as functions of x
    /// * `options` - Quadrature options
    fn dblquad<F, G, H>(
        &self,
        f: F,
        a: f64,
        b: f64,
        gfun: G,
        hfun: H,
        options: &NQuadOptions,
    ) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
        G: Fn(f64) -> f64,
        H: Fn(f64) -> f64;

    /// N-dimensional adaptive quadrature.
    ///
    /// Computes n-dimensional integrals using nested adaptive quadrature.
    /// For dimensions > 3, consider using Monte Carlo or QMC methods instead.
    ///
    /// # Arguments
    /// * `f` - Integrand function taking tensor of shape [n_points, n_dims]
    /// * `bounds` - Integration bounds for each dimension [(a1, b1), ...]
    /// * `options` - Quadrature options
    fn nquad<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &NQuadOptions,
    ) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Stiff ODE Solvers
    // ========================================================================

    /// Solve a stiff ODE using BDF (Backward Differentiation Formula) method.
    ///
    /// BDF is an implicit multistep method ideal for stiff problems where
    /// explicit methods would require impractically small time steps.
    /// Uses Newton iteration with **automatic Jacobian computation** via autograd.
    ///
    /// # Unique Capability
    ///
    /// This is the only Rust ODE solver with automatic Jacobian computation.
    /// The ODE function uses `DualTensor` and `dual_*` operations from
    /// `numr::autograd::dual_ops`, enabling exact Jacobians without finite differences.
    ///
    /// # Arguments
    /// * `f` - Right-hand side function using DualTensor ops: f(t, y, client) -> dy/dt
    /// * `t_span` - Integration interval [t0, tf]
    /// * `y0` - Initial condition
    /// * `options` - General ODE options
    /// * `bdf_options` - BDF-specific options (order, Newton parameters, sparse Jacobian)
    fn solve_ivp_bdf<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        options: &ODEOptions,
        bdf_options: &BDFOptions<R>,
    ) -> IntegrateResult<ODEResultTensor<R>>
    where
        F: Fn(&DualTensor<R>, &DualTensor<R>, &Self) -> Result<DualTensor<R>>;

    /// Solve a stiff ODE using Radau IIA method.
    ///
    /// Radau IIA is a 3-stage implicit Runge-Kutta method of order 5.
    /// More stable than BDF for extremely stiff problems.
    /// Uses **automatic Jacobian computation** via autograd.
    ///
    /// # Arguments
    /// * `f` - Right-hand side function using DualTensor ops: f(t, y, client) -> dy/dt
    /// * `t_span` - Integration interval [t0, tf]
    /// * `y0` - Initial condition
    /// * `options` - General ODE options
    /// * `radau_options` - Radau-specific options (Newton parameters, sparse Jacobian)
    fn solve_ivp_radau<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        options: &ODEOptions,
        radau_options: &RadauOptions<R>,
    ) -> IntegrateResult<ODEResultTensor<R>>
    where
        F: Fn(&DualTensor<R>, &DualTensor<R>, &Self) -> Result<DualTensor<R>>;

    /// Solve an ODE with automatic stiff/non-stiff method switching (LSODA).
    ///
    /// LSODA automatically detects stiffness and switches between
    /// Adams-Moulton (non-stiff) and BDF (stiff) methods.
    /// Uses **automatic Jacobian computation** via autograd when in stiff mode.
    ///
    /// # Arguments
    /// * `f` - Right-hand side function using DualTensor ops: f(t, y, client) -> dy/dt
    /// * `t_span` - Integration interval [t0, tf]
    /// * `y0` - Initial condition
    /// * `options` - General ODE options
    /// * `lsoda_options` - LSODA-specific options (switching thresholds)
    fn solve_ivp_lsoda<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        options: &ODEOptions,
        lsoda_options: &LSODAOptions,
    ) -> IntegrateResult<ODEResultTensor<R>>
    where
        F: Fn(&DualTensor<R>, &DualTensor<R>, &Self) -> Result<DualTensor<R>>;

    // ========================================================================
    // Boundary Value Problem Solver
    // ========================================================================

    /// Solve a two-point boundary value problem using collocation.
    ///
    /// Solves the system dy/dx = f(x, y) with boundary conditions bc(y(a), y(b)) = 0.
    ///
    /// # Arguments
    /// * `f` - ODE right-hand side f(x, y) where x is scalar tensor, y is state tensor
    /// * `bc` - Boundary condition function bc(ya, yb) returning residual tensor
    /// * `x` - Initial mesh points (1-D tensor)
    /// * `y` - Initial guess for solution at mesh points (shape [n_vars, n_points])
    /// * `options` - BVP solver options
    ///
    /// # Returns
    /// A [`BVPResult`] with the solution on the (possibly refined) mesh.
    fn solve_bvp<F, BC>(
        &self,
        f: F,
        bc: BC,
        x: &Tensor<R>,
        y: &Tensor<R>,
        options: &BVPOptions,
    ) -> IntegrateResult<BVPResult<R>>
    where
        F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
        BC: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Symplectic Integrators
    // ========================================================================

    /// Störmer-Verlet symplectic integrator for Hamiltonian systems.
    ///
    /// Solves Hamilton's equations:
    ///   dq/dt = ∂H/∂p = p/m  (velocity)
    ///   dp/dt = -∂H/∂q = F(q) (force)
    ///
    /// Preserves the symplectic structure, providing excellent long-term
    /// energy conservation.
    ///
    /// # Arguments
    /// * `force` - Force function F(q) returning -∂V/∂q
    /// * `t_span` - Integration interval [t0, tf]
    /// * `q0` - Initial positions
    /// * `p0` - Initial momenta
    /// * `options` - Symplectic integrator options (fixed step size)
    ///
    /// # Returns
    /// A [`SymplecticResult`] with position and momentum trajectories.
    fn verlet<F>(
        &self,
        force: F,
        t_span: [f64; 2],
        q0: &Tensor<R>,
        p0: &Tensor<R>,
        options: &SymplecticOptions,
    ) -> IntegrateResult<SymplecticResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Leapfrog symplectic integrator for Hamiltonian systems.
    ///
    /// Equivalent to Störmer-Verlet with a different arrangement of updates.
    /// Time-reversible and symplectic. Common in N-body simulations.
    ///
    /// # Arguments
    /// * `force` - Force function F(q) returning -∂V/∂q
    /// * `t_span` - Integration interval [t0, tf]
    /// * `q0` - Initial positions
    /// * `p0` - Initial momenta
    /// * `options` - Symplectic integrator options
    fn leapfrog<F>(
        &self,
        force: F,
        t_span: [f64; 2],
        q0: &Tensor<R>,
        p0: &Tensor<R>,
        options: &SymplecticOptions,
    ) -> IntegrateResult<SymplecticResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
