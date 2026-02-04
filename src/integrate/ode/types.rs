//! Types for ODE solvers.

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
pub struct BDFOptions {
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
}

impl Default for BDFOptions {
    fn default() -> Self {
        Self {
            max_order: 5,
            newton_tol: 1e-6,
            max_newton_iter: 10,
            numerical_jacobian: true,
        }
    }
}

impl BDFOptions {
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
}

/// Options specific to Radau IIA solver.
#[derive(Debug, Clone)]
pub struct RadauOptions {
    /// Newton iteration tolerance (default: 1e-6).
    pub newton_tol: f64,

    /// Maximum Newton iterations per step (default: 10).
    pub max_newton_iter: usize,

    /// Use simplified Newton (reuse Jacobian) (default: true).
    ///
    /// Simplified Newton reuses the Jacobian across iterations for efficiency.
    pub simplified_newton: bool,
}

impl Default for RadauOptions {
    fn default() -> Self {
        Self {
            newton_tol: 1e-6,
            max_newton_iter: 10,
            simplified_newton: true,
        }
    }
}

impl RadauOptions {
    /// Set Newton iteration parameters.
    pub fn newton_params(mut self, tol: f64, max_iter: usize) -> Self {
        self.newton_tol = tol;
        self.max_newton_iter = max_iter;
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
        let opts = BDFOptions::default();
        assert_eq!(opts.max_order, 5);
        assert_eq!(opts.max_newton_iter, 10);

        let opts = BDFOptions::default().max_order(3).newton_params(1e-8, 20);
        assert_eq!(opts.max_order, 3);
        assert_eq!(opts.newton_tol, 1e-8);
        assert_eq!(opts.max_newton_iter, 20);
    }

    #[test]
    fn test_radau_options() {
        let opts = RadauOptions::default();
        assert!(opts.simplified_newton);

        let opts = RadauOptions::default().newton_params(1e-10, 15);
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
}
