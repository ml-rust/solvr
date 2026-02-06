//! Types for ODE solvers and DAE solvers.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(feature = "sparse")]
use numr::sparse::CsrData;

#[cfg(feature = "sparse")]
pub use crate::integrate::impl_generic::ode::direct_solver_config::{
    DirectSolverConfig, SparseSolverStrategy,
};

// ============================================================================
// Sparse Jacobian Configuration
// ============================================================================

/// Configuration for sparse Jacobian solvers in implicit ODE methods.
///
/// For large-scale stiff systems (e.g., PDE discretizations with 10k+ variables),
/// dense linear algebra becomes infeasible (O(n²) memory, O(n³) time). Sparse
/// solvers exploit the structure of the Jacobian to achieve O(nnz) memory and
/// O(k·nnz) time complexity.
///
/// # Feature Flag
///
/// Requires the `sparse` feature to be enabled in both `numr` and `solvr`.
///
/// # When to Use
///
/// Enable sparse mode when:
/// - System size n > 1000
/// - Jacobian has sparse structure (PDE discretizations, chemical networks)
/// - Dense solve causes OOM or is too slow
///
/// # Example
///
/// ```ignore
/// use solvr::integrate::ode::SparseJacobianConfig;
/// use numr::algorithm::iterative::PreconditionerType;
///
/// let sparse_config = SparseJacobianConfig {
///     enabled: true,
///     pattern: None,  // Auto-detect or provide CsrData
///     gmres_tol: 1e-10,
///     max_gmres_iter: 100,
///     preconditioner: PreconditionerType::Ilu0,
///     _phantom: std::marker::PhantomData,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SparseJacobianConfig<R: Runtime> {
    /// Enable sparse Jacobian solver (default: false).
    ///
    /// When enabled, uses GMRES instead of dense LU for Newton system solves.
    /// **Note**: Requires the `sparse` feature to be enabled.
    pub enabled: bool,

    /// Sparsity pattern of the Jacobian (optional).
    ///
    /// If provided, the dense Jacobian will be converted to CSR format using
    /// this pattern. If None, full dense matrix is used (defeats the purpose).
    ///
    /// **Note**: Automatic sparsity detection is not yet implemented.
    /// **Note**: Only available with the `sparse` feature.
    #[cfg(feature = "sparse")]
    pub pattern: Option<CsrData<R>>,

    /// GMRES convergence tolerance (default: 1e-10).
    ///
    /// Should match ODE accuracy requirements. Lower values increase Newton
    /// iteration accuracy but require more GMRES iterations.
    pub gmres_tol: f64,

    /// Maximum GMRES iterations (default: 100).
    pub max_gmres_iter: usize,

    /// Preconditioner type (default: Ilu0).
    ///
    /// ILU(0) is recommended for general non-symmetric Jacobians. IC(0) can
    /// be used if the Jacobian is symmetric positive definite.
    /// **Note**: Only available with the `sparse` feature.
    #[cfg(feature = "sparse")]
    pub preconditioner: numr::algorithm::iterative::PreconditionerType,

    /// Sparse solver strategy (default: Gmres).
    ///
    /// Controls whether the Newton system is solved using iterative GMRES
    /// or direct sparse LU factorization.
    /// **Note**: Only available with the `sparse` feature.
    #[cfg(feature = "sparse")]
    pub solver_strategy: SparseSolverStrategy,

    /// Configuration for direct sparse LU solver (default: defaults).
    ///
    /// Only used when `solver_strategy` is `DirectLU` or `Auto`.
    /// **Note**: Only available with the `sparse` feature.
    #[cfg(feature = "sparse")]
    pub direct_solver_config: DirectSolverConfig,

    /// Phantom data to preserve the Runtime type parameter.
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Runtime> Default for SparseJacobianConfig<R> {
    fn default() -> Self {
        Self {
            enabled: false,
            #[cfg(feature = "sparse")]
            pattern: None,
            gmres_tol: 1e-10,
            max_gmres_iter: 100,
            #[cfg(feature = "sparse")]
            preconditioner: numr::algorithm::iterative::PreconditionerType::Ilu0,
            #[cfg(feature = "sparse")]
            solver_strategy: SparseSolverStrategy::default(),
            #[cfg(feature = "sparse")]
            direct_solver_config: DirectSolverConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> SparseJacobianConfig<R> {
    /// Create a disabled sparse configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Enable sparse mode with default settings.
    ///
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Set the sparsity pattern.
    ///
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn with_pattern(mut self, pattern: CsrData<R>) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set GMRES tolerance.
    pub fn with_gmres_tol(mut self, tol: f64) -> Self {
        self.gmres_tol = tol;
        self
    }

    /// Set preconditioner type.
    ///
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn with_preconditioner(
        mut self,
        precond: numr::algorithm::iterative::PreconditionerType,
    ) -> Self {
        self.preconditioner = precond;
        self
    }

    /// Set the sparse solver strategy.
    ///
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn with_solver_strategy(mut self, strategy: SparseSolverStrategy) -> Self {
        self.solver_strategy = strategy;
        self
    }

    /// Configure for direct sparse LU solver.
    ///
    /// Convenience method that enables sparse mode and sets DirectLU strategy.
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn with_direct_lu() -> Self {
        Self {
            enabled: true,
            solver_strategy: SparseSolverStrategy::DirectLU,
            ..Default::default()
        }
    }

    /// Set direct solver configuration.
    ///
    /// **Note**: Requires the `sparse` feature to be enabled.
    #[cfg(feature = "sparse")]
    pub fn with_direct_solver_config(mut self, config: DirectSolverConfig) -> Self {
        self.direct_solver_config = config;
        self
    }
}

/// ODE solver method.
///
/// # Available Methods
///
/// ## Explicit Methods (Non-Stiff Problems)
///
/// | Method | Order | Stages | Use Case |
/// |--------|-------|--------|----------|
/// | RK23   | 2(3)  | 4      | Fast, lower accuracy |
/// | RK45   | 4(5)  | 6      | General purpose (recommended) |
/// | DOP853 | 8(5,3)| 12     | High accuracy requirements |
///
/// ## Implicit Methods (Stiff Problems)
///
/// | Method | Order | Use Case |
/// |--------|-------|----------|
/// | BDF    | 1-5   | Stiff problems, chemical kinetics |
/// | Radau  | 5     | Very stiff problems |
/// | LSODA  | auto  | Unknown stiffness (auto-switches) |
///
/// ## Symplectic Methods (Hamiltonian Systems)
///
/// | Method   | Order | Use Case |
/// |----------|-------|----------|
/// | Verlet   | 2     | Molecular dynamics, energy conservation |
/// | Leapfrog | 2     | N-body simulations |
///
/// # Choosing a Method
///
/// - **RK23**: Use when speed is more important than accuracy, or for getting
///   a rough initial estimate.
/// - **RK45**: The default choice. Works well for most non-stiff problems.
/// - **DOP853**: Use for high-accuracy requirements on smooth problems.
/// - **BDF**: Use for stiff problems where explicit methods require tiny steps.
/// - **Radau**: Use for very stiff problems (e.g., chemical kinetics).
/// - **LSODA**: Use when you don't know if the problem is stiff.
/// - **Verlet/Leapfrog**: Use for Hamiltonian systems requiring energy conservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ODEMethod {
    /// Bogacki-Shampine 2(3) - low accuracy, fast.
    ///
    /// 4 stages per step. Good for problems where speed matters more than
    /// precision, or for getting a rough initial solution.
    RK23,

    /// Dormand-Prince 4(5) - general purpose (default).
    ///
    /// 6 stages per step. Recommended for most problems. Good balance of
    /// accuracy and speed. Uses FSAL (First Same As Last) property for
    /// efficiency.
    #[default]
    RK45,

    /// Dormand-Prince 8(5,3) - high accuracy.
    ///
    /// 12 stages per step. An 8th order method with embedded 5th order error
    /// estimator. Best for high-accuracy requirements on smooth problems.
    /// Takes larger steps than RK45 for tight tolerances, compensating for
    /// the additional work per step.
    DOP853,

    /// Backward Differentiation Formula (BDF) - implicit, stiff problems.
    ///
    /// Variable order (1-5). Uses Newton iteration to solve implicit equations.
    /// Excellent for stiff ODEs where explicit methods would require tiny steps.
    BDF,

    /// Radau IIA order 5 - implicit Runge-Kutta for very stiff problems.
    ///
    /// 3-stage implicit method. More stable than BDF for extremely stiff problems.
    /// Uses Newton iteration with automatic Jacobian computation.
    Radau,

    /// LSODA - automatic stiff/non-stiff switching.
    ///
    /// Automatically switches between Adams-Moulton (non-stiff) and BDF (stiff)
    /// based on detected stiffness. Use when you don't know if the problem is stiff.
    LSODA,

    /// Störmer-Verlet - symplectic integrator for Hamiltonian systems.
    ///
    /// 2nd order. Conserves energy over long integrations. Use for molecular
    /// dynamics, planetary motion, and other conservative systems.
    Verlet,

    /// Leapfrog - symplectic integrator for Hamiltonian systems.
    ///
    /// 2nd order, time-reversible. Equivalent to Verlet with different variable
    /// arrangement. Common in N-body simulations.
    Leapfrog,
}

impl ODEMethod {
    /// Get the order of the method.
    pub fn order(&self) -> usize {
        match self {
            Self::RK23 => 3,
            Self::RK45 => 5,
            Self::DOP853 => 8,
            Self::BDF => 5, // Max order
            Self::Radau => 5,
            Self::LSODA => 5, // Max order (variable)
            Self::Verlet => 2,
            Self::Leapfrog => 2,
        }
    }

    /// Get the error estimator order.
    pub fn error_order(&self) -> usize {
        match self {
            Self::RK23 => 2,
            Self::RK45 => 4,
            Self::DOP853 => 5,
            Self::BDF => 4,   // Embedded error estimate
            Self::Radau => 3, // Embedded lower-order
            Self::LSODA => 4,
            Self::Verlet => 2,
            Self::Leapfrog => 2,
        }
    }

    /// Returns true if this method is implicit (requires solving nonlinear equations).
    pub fn is_implicit(&self) -> bool {
        matches!(self, Self::BDF | Self::Radau | Self::LSODA)
    }

    /// Returns true if this method is symplectic (preserves Hamiltonian structure).
    pub fn is_symplectic(&self) -> bool {
        matches!(self, Self::Verlet | Self::Leapfrog)
    }

    /// Returns true if this method is suitable for stiff problems.
    pub fn is_stiff_solver(&self) -> bool {
        matches!(self, Self::BDF | Self::Radau | Self::LSODA)
    }
}

/// Options for ODE solvers.
#[derive(Debug, Clone)]
pub struct ODEOptions {
    /// Solver method (default: RK45)
    pub method: ODEMethod,

    /// Relative tolerance (default: 1e-3)
    pub rtol: f64,

    /// Absolute tolerance (default: 1e-6)
    pub atol: f64,

    /// Initial step size (default: auto-computed)
    pub h0: Option<f64>,

    /// Maximum step size (default: unbounded)
    pub max_step: Option<f64>,

    /// Minimum step size (default: machine epsilon)
    pub min_step: Option<f64>,

    /// Maximum number of steps (default: 10000)
    pub max_steps: usize,

    /// Dense output - evaluate solution at any point (default: false)
    pub dense_output: bool,
}

impl Default for ODEOptions {
    fn default() -> Self {
        Self {
            method: ODEMethod::default(),
            rtol: 1e-3,
            atol: 1e-6,
            h0: None,
            max_step: None,
            min_step: None,
            max_steps: 10000,
            dense_output: false,
        }
    }
}

impl ODEOptions {
    /// Create options with specified tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }

    /// Create options with specified method.
    pub fn with_method(method: ODEMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Set the method.
    pub fn method(mut self, method: ODEMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the tolerances.
    pub fn tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Set the initial step size.
    pub fn initial_step(mut self, h0: f64) -> Self {
        self.h0 = Some(h0);
        self
    }

    /// Set step size bounds.
    pub fn step_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_step = Some(min);
        self.max_step = Some(max);
        self
    }

    /// Set maximum number of steps.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }
}

/// Options specific to BDF (Backward Differentiation Formula) solver.
#[derive(Debug, Clone)]
pub struct BDFOptions<R: Runtime> {
    /// Maximum BDF order (1-5, default: 5).
    ///
    /// Higher orders are more accurate but may be less stable for very stiff problems.
    pub max_order: usize,

    /// Newton iteration tolerance (default: 1e-6).
    ///
    /// Controls convergence of the Newton solver for implicit equations.
    pub newton_tol: f64,

    /// Maximum Newton iterations per step (default: 10).
    pub max_newton_iter: usize,

    /// Use numerical Jacobian (default: true).
    ///
    /// If false, assumes Jacobian is provided analytically.
    pub numerical_jacobian: bool,

    /// Sparse Jacobian configuration (default: disabled).
    ///
    /// Enable for large-scale systems (n > 1000) with sparse Jacobians.
    pub sparse_jacobian: SparseJacobianConfig<R>,
}

impl<R: Runtime> Default for BDFOptions<R> {
    fn default() -> Self {
        Self {
            max_order: 5,
            newton_tol: 1e-6,
            max_newton_iter: 10,
            numerical_jacobian: true,
            sparse_jacobian: SparseJacobianConfig::default(),
        }
    }
}

impl<R: Runtime> BDFOptions<R> {
    /// Set maximum order.
    pub fn max_order(mut self, order: usize) -> Self {
        self.max_order = order.clamp(1, 5);
        self
    }

    /// Set Newton iteration parameters.
    pub fn newton_params(mut self, tol: f64, max_iter: usize) -> Self {
        self.newton_tol = tol;
        self.max_newton_iter = max_iter;
        self
    }

    /// Enable sparse Jacobian solver.
    pub fn with_sparse_jacobian(mut self, config: SparseJacobianConfig<R>) -> Self {
        self.sparse_jacobian = config;
        self
    }
}

/// Options specific to Radau IIA solver.
#[derive(Debug, Clone)]
pub struct RadauOptions<R: Runtime> {
    /// Newton iteration tolerance (default: 1e-6).
    pub newton_tol: f64,

    /// Maximum Newton iterations per step (default: 10).
    pub max_newton_iter: usize,

    /// Use simplified Newton (reuse Jacobian) (default: true).
    ///
    /// Simplified Newton reuses the Jacobian across iterations for efficiency.
    pub simplified_newton: bool,

    /// Sparse Jacobian configuration (default: disabled).
    ///
    /// Enable for large-scale systems (n > 1000) with sparse Jacobians.
    pub sparse_jacobian: SparseJacobianConfig<R>,
}

impl<R: Runtime> Default for RadauOptions<R> {
    fn default() -> Self {
        Self {
            newton_tol: 1e-6,
            max_newton_iter: 10,
            simplified_newton: true,
            sparse_jacobian: SparseJacobianConfig::default(),
        }
    }
}

impl<R: Runtime> RadauOptions<R> {
    /// Set Newton iteration parameters.
    pub fn newton_params(mut self, tol: f64, max_iter: usize) -> Self {
        self.newton_tol = tol;
        self.max_newton_iter = max_iter;
        self
    }

    /// Enable sparse Jacobian solver.
    pub fn with_sparse_jacobian(mut self, config: SparseJacobianConfig<R>) -> Self {
        self.sparse_jacobian = config;
        self
    }
}

/// Options specific to LSODA solver.
#[derive(Debug, Clone)]
pub struct LSODAOptions {
    /// Number of step rejections before switching to BDF (stiff mode).
    ///
    /// Default: 3. Lower values switch to stiff mode more aggressively.
    pub stiff_threshold: usize,

    /// Number of successful steps before switching back to Adams.
    ///
    /// Default: 10. Higher values keep using BDF longer after detecting stiffness.
    pub nonstiff_threshold: usize,

    /// Maximum order for Adams-Moulton (non-stiff) method.
    ///
    /// Default: 12. Valid range: 1-12.
    pub max_adams_order: usize,

    /// Maximum order for BDF (stiff) method.
    ///
    /// Default: 5. Valid range: 1-5.
    pub max_bdf_order: usize,
}

impl Default for LSODAOptions {
    fn default() -> Self {
        Self {
            stiff_threshold: 3,
            nonstiff_threshold: 10,
            max_adams_order: 12,
            max_bdf_order: 5,
        }
    }
}

/// Options specific to symplectic integrators (Verlet, Leapfrog).
#[derive(Debug, Clone)]
pub struct SymplecticOptions {
    /// Fixed step size for integration.
    ///
    /// Symplectic integrators typically use fixed steps to preserve
    /// geometric properties.
    pub dt: f64,

    /// Number of steps (computed from dt and t_span if not set).
    pub n_steps: Option<usize>,
}

impl Default for SymplecticOptions {
    fn default() -> Self {
        Self {
            dt: 0.01,
            n_steps: None,
        }
    }
}

impl SymplecticOptions {
    /// Create options with specified step size.
    pub fn with_dt(dt: f64) -> Self {
        Self { dt, n_steps: None }
    }

    /// Create options with specified number of steps.
    pub fn with_n_steps(n_steps: usize) -> Self {
        Self {
            dt: 0.0, // Will be computed from t_span
            n_steps: Some(n_steps),
        }
    }
}

/// Options for boundary value problem (BVP) solvers.
#[derive(Debug, Clone)]
pub struct BVPOptions {
    /// Relative tolerance for solution (default: 1e-3).
    pub rtol: f64,

    /// Absolute tolerance for solution (default: 1e-6).
    pub atol: f64,

    /// Maximum iterations for nonlinear solver (default: 100).
    pub max_iter: usize,

    /// Initial mesh size (number of points) (default: 10).
    pub initial_mesh_size: usize,

    /// Maximum mesh size after refinement (default: 1000).
    pub max_mesh_size: usize,
}

impl Default for BVPOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-3,
            atol: 1e-6,
            max_iter: 100,
            initial_mesh_size: 10,
            max_mesh_size: 1000,
        }
    }
}

impl BVPOptions {
    /// Create options with specified tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }

    /// Set mesh parameters.
    pub fn mesh_params(mut self, initial: usize, max: usize) -> Self {
        self.initial_mesh_size = initial;
        self.max_mesh_size = max;
        self
    }
}

// ============================================================================
// DAE (Differential-Algebraic Equation) Types
// ============================================================================

/// Classification of variables in a DAE system.
///
/// For a DAE F(t, y, y') = 0, each component of y is either:
/// - **Differential**: Appears in y' (has time derivative)
/// - **Algebraic**: No y' term (represents a constraint)
///
/// This classification is used for:
/// - Consistent initial condition computation
/// - Error estimation (optionally excluding algebraic variables)
/// - Scaling in Newton iteration
///
/// # Example
///
/// For a pendulum in Cartesian coordinates:
/// ```ignore
/// // x'' = λ·x, y'' = λ·y - g, x² + y² = L²
/// // As first-order system: [x, y, vx, vy, λ]
/// let var_types = vec![
///     DAEVariableType::Differential, // x
///     DAEVariableType::Differential, // y
///     DAEVariableType::Differential, // vx
///     DAEVariableType::Differential, // vy
///     DAEVariableType::Algebraic,    // λ (Lagrange multiplier)
/// ];
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DAEVariableType {
    /// Variable appears in y' (has time derivative).
    Differential,
    /// Variable has no y' term (constraint equation).
    Algebraic,
}

/// Options for DAE (Differential-Algebraic Equation) solvers.
///
/// DAEs are of the form F(t, y, y') = 0, which generalizes ODEs to include
/// algebraic constraints. This struct configures the BDF-based implicit solver.
///
/// # Example
///
/// ```ignore
/// use solvr::integrate::ode::{DAEOptions, DAEVariableType};
///
/// let dae_opts = DAEOptions::default()
///     .with_variable_types(vec![
///         DAEVariableType::Differential,
///         DAEVariableType::Algebraic,
///     ])
///     .with_ic_tolerance(1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct DAEOptions<R: Runtime> {
    /// Classification of each variable (differential vs algebraic).
    ///
    /// If `None`, all variables are treated as differential.
    /// Providing this enables better initial condition computation
    /// and can improve error estimation accuracy.
    pub variable_types: Option<Vec<DAEVariableType>>,

    /// Tolerance for consistent initial condition computation (default: 1e-10).
    ///
    /// The solver refines (y0, yp0) to satisfy F(t0, y0, yp0) ≈ 0
    /// within this tolerance.
    pub ic_tol: f64,

    /// Maximum iterations for initial condition refinement (default: 20).
    pub max_ic_iter: usize,

    /// Newton iteration tolerance (default: 1e-6).
    ///
    /// Controls convergence of Newton solver for implicit BDF equations.
    pub newton_tol: f64,

    /// Maximum Newton iterations per time step (default: 10).
    pub max_newton_iter: usize,

    /// Maximum BDF order (1-5, default: 5).
    ///
    /// Higher orders are more accurate but may be less stable
    /// for very stiff or high-index DAEs.
    pub max_order: usize,

    /// Exclude algebraic variables from error estimation (default: true).
    ///
    /// Following SUNDIALS IDA convention, algebraic variables are typically
    /// excluded from the local error test since their values are determined
    /// by the constraints rather than by integration.
    pub exclude_algebraic_from_error: bool,

    /// Whether to return y' trajectory in results (default: false).
    ///
    /// Enable if you need derivative values at each time step.
    pub return_yp: bool,

    /// Sparse Jacobian configuration (default: disabled).
    ///
    /// Enable for large-scale systems (n > 1000) with sparse structure.
    pub sparse_jacobian: SparseJacobianConfig<R>,
}

impl<R: Runtime> Default for DAEOptions<R> {
    fn default() -> Self {
        Self {
            variable_types: None,
            ic_tol: 1e-10,
            max_ic_iter: 20,
            newton_tol: 1e-6,
            max_newton_iter: 10,
            max_order: 5,
            exclude_algebraic_from_error: true,
            return_yp: false,
            sparse_jacobian: SparseJacobianConfig::default(),
        }
    }
}

impl<R: Runtime> DAEOptions<R> {
    /// Set variable type classification.
    pub fn with_variable_types(mut self, types: Vec<DAEVariableType>) -> Self {
        self.variable_types = Some(types);
        self
    }

    /// Set initial condition tolerance.
    pub fn with_ic_tolerance(mut self, tol: f64) -> Self {
        self.ic_tol = tol;
        self
    }

    /// Set Newton iteration parameters.
    pub fn with_newton_params(mut self, tol: f64, max_iter: usize) -> Self {
        self.newton_tol = tol;
        self.max_newton_iter = max_iter;
        self
    }

    /// Set maximum BDF order.
    pub fn with_max_order(mut self, order: usize) -> Self {
        self.max_order = order.clamp(1, 5);
        self
    }

    /// Set whether to exclude algebraic variables from error estimation.
    pub fn with_exclude_algebraic(mut self, exclude: bool) -> Self {
        self.exclude_algebraic_from_error = exclude;
        self
    }

    /// Enable returning y' trajectory.
    pub fn with_return_yp(mut self, return_yp: bool) -> Self {
        self.return_yp = return_yp;
        self
    }

    /// Set sparse Jacobian configuration.
    pub fn with_sparse_jacobian(mut self, config: SparseJacobianConfig<R>) -> Self {
        self.sparse_jacobian = config;
        self
    }
}

/// Result of DAE integration.
///
/// Contains the solution trajectory and optional derivative trajectory.
#[derive(Debug, Clone)]
pub struct DAEResultTensor<R: Runtime> {
    /// Time points where solution was computed (1-D tensor).
    pub t: Tensor<R>,

    /// Solution values - shape [n_steps, n_vars].
    pub y: Tensor<R>,

    /// Derivative values - shape [n_steps, n_vars] (if requested).
    pub yp: Option<Tensor<R>>,

    /// Whether integration was successful.
    pub success: bool,

    /// Status message (e.g., why integration failed).
    pub message: Option<String>,

    /// Number of residual function evaluations.
    pub nfev: usize,

    /// Number of Jacobian evaluations.
    pub njac: usize,

    /// Number of Newton iterations for initial conditions.
    pub n_ic_iter: usize,

    /// Number of accepted time steps.
    pub naccept: usize,

    /// Number of rejected time steps.
    pub nreject: usize,
}

impl<R: Runtime> DAEResultTensor<R> {
    /// Get the final state as a Vec<f64>.
    pub fn y_final_vec(&self) -> Vec<f64> {
        let shape = self.y.shape();
        if shape.len() != 2 || shape[0] == 0 {
            return vec![];
        }
        let n_steps = shape[0];
        let n_vars = shape[1];
        let all_data: Vec<f64> = self.y.to_vec();
        let last_row_start = (n_steps - 1) * n_vars;
        all_data[last_row_start..].to_vec()
    }

    /// Get the final derivative as a Vec<f64> (if available).
    pub fn yp_final_vec(&self) -> Option<Vec<f64>> {
        self.yp.as_ref().map(|yp| {
            let shape = yp.shape();
            if shape.len() != 2 || shape[0] == 0 {
                return vec![];
            }
            let n_steps = shape[0];
            let n_vars = shape[1];
            let all_data: Vec<f64> = yp.to_vec();
            let last_row_start = (n_steps - 1) * n_vars;
            all_data[last_row_start..].to_vec()
        })
    }
}

// ============================================================================
// Event Handling Types
// ============================================================================

/// Direction for event detection.
///
/// Specifies which zero-crossing direction should trigger the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventDirection {
    /// Trigger on any zero-crossing (both increasing and decreasing).
    #[default]
    Any,

    /// Trigger only when the event function crosses from negative to positive.
    Increasing,

    /// Trigger only when the event function crosses from positive to negative.
    Decreasing,
}

/// Specification for an event function.
///
/// Defines behavior of a single event function g(t, y) = 0.
#[derive(Debug, Clone)]
pub struct EventSpec {
    /// If true, stop integration when this event occurs.
    pub terminal: bool,

    /// Which direction of zero-crossing to detect.
    pub direction: EventDirection,
}

impl Default for EventSpec {
    fn default() -> Self {
        Self {
            terminal: false,
            direction: EventDirection::Any,
        }
    }
}

impl EventSpec {
    /// Create a terminal event (stops integration).
    pub fn terminal() -> Self {
        Self {
            terminal: true,
            direction: EventDirection::Any,
        }
    }

    /// Create a non-terminal event (records but continues).
    pub fn non_terminal() -> Self {
        Self {
            terminal: false,
            direction: EventDirection::Any,
        }
    }

    /// Set the detection direction.
    pub fn direction(mut self, dir: EventDirection) -> Self {
        self.direction = dir;
        self
    }
}

/// Record of a detected event.
#[derive(Debug, Clone)]
pub struct EventRecord<R: Runtime> {
    /// Time at which the event occurred.
    pub t: f64,

    /// State at the event time.
    pub y: Tensor<R>,

    /// Index of the event function that triggered (0-indexed).
    pub event_index: usize,

    /// Value of the event function at the event (should be near zero).
    pub event_value: f64,
}

/// Options for event detection and root finding.
#[derive(Debug, Clone)]
pub struct EventOptions {
    /// Tolerance for root finding (default: 1e-10).
    pub root_tol: f64,

    /// Maximum iterations for root refinement (default: 100).
    pub max_root_iter: usize,
}

impl Default for EventOptions {
    fn default() -> Self {
        Self {
            root_tol: 1e-10,
            max_root_iter: 100,
        }
    }
}

impl EventOptions {
    /// Set root finding tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.root_tol = tol;
        self
    }

    /// Set maximum root finding iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_root_iter = max_iter;
        self
    }
}

/// Result of ODE integration with event detection.
#[derive(Debug, Clone)]
pub struct ODEResultWithEvents<R: Runtime> {
    /// Time points where solution was computed (1-D tensor).
    pub t: Tensor<R>,

    /// Solution values - shape [n_steps, n_vars].
    pub y: Tensor<R>,

    /// Whether integration was successful.
    pub success: bool,

    /// Status message.
    pub message: Option<String>,

    /// Number of function evaluations.
    pub nfev: usize,

    /// Number of accepted steps.
    pub naccept: usize,

    /// Number of rejected steps.
    pub nreject: usize,

    /// Method used for integration.
    pub method: ODEMethod,

    /// Detected events in chronological order.
    pub events: Vec<EventRecord<R>>,

    /// Whether integration was terminated by an event.
    pub terminated_by_event: bool,

    /// Index of the terminal event (if terminated_by_event is true).
    pub terminal_event_index: Option<usize>,
}

impl<R: Runtime> ODEResultWithEvents<R> {
    /// Get the final state as a Vec<f64>.
    pub fn y_final_vec(&self) -> Vec<f64> {
        let shape = self.y.shape();
        if shape.len() != 2 || shape[0] == 0 {
            return vec![];
        }

        let n_steps = shape[0];
        let n_vars = shape[1];
        let all_data: Vec<f64> = self.y.to_vec();
        let last_row_start = (n_steps - 1) * n_vars;
        all_data[last_row_start..].to_vec()
    }

    /// Get all events for a specific event function index.
    pub fn events_for(&self, event_index: usize) -> Vec<&EventRecord<R>> {
        self.events
            .iter()
            .filter(|e| e.event_index == event_index)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_method() {
        assert_eq!(ODEMethod::RK23.order(), 3);
        assert_eq!(ODEMethod::RK23.error_order(), 2);
        assert_eq!(ODEMethod::RK45.order(), 5);
        assert_eq!(ODEMethod::RK45.error_order(), 4);
        assert_eq!(ODEMethod::DOP853.order(), 8);
        assert_eq!(ODEMethod::DOP853.error_order(), 5);
    }

    #[test]
    fn test_stiff_methods() {
        assert!(ODEMethod::BDF.is_implicit());
        assert!(ODEMethod::BDF.is_stiff_solver());
        assert!(!ODEMethod::BDF.is_symplectic());

        assert!(ODEMethod::Radau.is_implicit());
        assert!(ODEMethod::LSODA.is_stiff_solver());
    }

    #[test]
    fn test_symplectic_methods() {
        assert!(ODEMethod::Verlet.is_symplectic());
        assert!(ODEMethod::Leapfrog.is_symplectic());
        assert!(!ODEMethod::Verlet.is_implicit());
        assert!(!ODEMethod::RK45.is_symplectic());
    }

    #[test]
    fn test_ode_options() {
        let opts = ODEOptions::default();
        assert_eq!(opts.method, ODEMethod::RK45);
        assert_eq!(opts.rtol, 1e-3);
        assert_eq!(opts.atol, 1e-6);

        let opts = ODEOptions::with_tolerances(1e-6, 1e-9);
        assert_eq!(opts.rtol, 1e-6);
        assert_eq!(opts.atol, 1e-9);
    }

    #[test]
    fn test_bdf_options() {
        use numr::runtime::cpu::CpuRuntime;

        let opts = BDFOptions::<CpuRuntime>::default();
        assert_eq!(opts.max_order, 5);
        assert_eq!(opts.max_newton_iter, 10);
        assert!(!opts.sparse_jacobian.enabled);

        let opts = BDFOptions::<CpuRuntime>::default()
            .max_order(3)
            .newton_params(1e-8, 20);
        assert_eq!(opts.max_order, 3);
        assert_eq!(opts.newton_tol, 1e-8);
        assert_eq!(opts.max_newton_iter, 20);
    }

    #[test]
    fn test_radau_options() {
        use numr::runtime::cpu::CpuRuntime;

        let opts = RadauOptions::<CpuRuntime>::default();
        assert!(opts.simplified_newton);
        assert!(!opts.sparse_jacobian.enabled);

        let opts = RadauOptions::<CpuRuntime>::default().newton_params(1e-10, 15);
        assert_eq!(opts.newton_tol, 1e-10);
        assert_eq!(opts.max_newton_iter, 15);
    }

    #[test]
    fn test_lsoda_options() {
        let opts = LSODAOptions::default();
        assert_eq!(opts.stiff_threshold, 3);
        assert_eq!(opts.nonstiff_threshold, 10);
        assert_eq!(opts.max_adams_order, 12);
        assert_eq!(opts.max_bdf_order, 5);
    }

    #[test]
    fn test_symplectic_options() {
        let opts = SymplecticOptions::default();
        assert_eq!(opts.dt, 0.01);
        assert!(opts.n_steps.is_none());

        let opts = SymplecticOptions::with_dt(0.001);
        assert_eq!(opts.dt, 0.001);

        let opts = SymplecticOptions::with_n_steps(1000);
        assert_eq!(opts.n_steps, Some(1000));
    }

    #[test]
    fn test_bvp_options() {
        let opts = BVPOptions::default();
        assert_eq!(opts.initial_mesh_size, 10);
        assert_eq!(opts.max_mesh_size, 1000);

        let opts = BVPOptions::with_tolerances(1e-6, 1e-9).mesh_params(20, 500);
        assert_eq!(opts.rtol, 1e-6);
        assert_eq!(opts.initial_mesh_size, 20);
        assert_eq!(opts.max_mesh_size, 500);
    }

    #[test]
    fn test_dae_variable_type() {
        let diff = DAEVariableType::Differential;
        let alg = DAEVariableType::Algebraic;
        assert_ne!(diff, alg);
        assert_eq!(diff, DAEVariableType::Differential);
    }

    #[test]
    fn test_dae_options() {
        use numr::runtime::cpu::CpuRuntime;

        let opts = DAEOptions::<CpuRuntime>::default();
        assert!(opts.variable_types.is_none());
        assert_eq!(opts.ic_tol, 1e-10);
        assert_eq!(opts.max_ic_iter, 20);
        assert_eq!(opts.newton_tol, 1e-6);
        assert_eq!(opts.max_order, 5);
        assert!(opts.exclude_algebraic_from_error);
        assert!(!opts.return_yp);

        let opts = DAEOptions::<CpuRuntime>::default()
            .with_variable_types(vec![
                DAEVariableType::Differential,
                DAEVariableType::Algebraic,
            ])
            .with_ic_tolerance(1e-12)
            .with_newton_params(1e-8, 15)
            .with_max_order(3)
            .with_exclude_algebraic(false)
            .with_return_yp(true);

        assert!(opts.variable_types.is_some());
        assert_eq!(opts.variable_types.as_ref().unwrap().len(), 2);
        assert_eq!(opts.ic_tol, 1e-12);
        assert_eq!(opts.newton_tol, 1e-8);
        assert_eq!(opts.max_newton_iter, 15);
        assert_eq!(opts.max_order, 3);
        assert!(!opts.exclude_algebraic_from_error);
        assert!(opts.return_yp);
    }
}
