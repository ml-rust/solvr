//! Types for integration algorithms.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of subdivisions (default: 50)
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            limit: 50,
        }
    }
}

/// Result of adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadResult<R: Runtime> {
    /// Computed integral value (0-D tensor)
    pub integral: Tensor<R>,
    /// Estimated absolute error
    pub error: f64,
    /// Number of function evaluations
    pub neval: usize,
    /// Whether integration converged
    pub converged: bool,
}

/// Options for Romberg integration.
#[derive(Debug, Clone)]
pub struct RombergOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of extrapolation levels (default: 20)
    pub max_levels: usize,
}

impl Default for RombergOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            max_levels: 20,
        }
    }
}

// ============================================================================
// Tanh-Sinh (Double Exponential) Quadrature
// ============================================================================

/// Options for tanh-sinh (double exponential) quadrature.
///
/// This method is particularly effective for integrals with endpoint
/// singularities or infinite derivatives at the boundaries.
#[derive(Debug, Clone)]
pub struct TanhSinhOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum refinement levels (default: 10)
    ///
    /// Each level doubles the number of quadrature points.
    pub max_levels: usize,
}

impl Default for TanhSinhOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            max_levels: 10,
        }
    }
}

impl TanhSinhOptions {
    /// Create options with specified tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }
}

// ============================================================================
// Monte Carlo Integration
// ============================================================================

/// Method for Monte Carlo integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonteCarloMethod {
    /// Plain Monte Carlo - simple random sampling.
    #[default]
    Plain,

    /// Stratified sampling - divide domain into strata, sample each.
    ///
    /// Reduces variance when function varies smoothly across strata.
    Stratified {
        /// Number of strata per dimension.
        n_strata: usize,
    },

    /// Antithetic variates - use x and (1-x) pairs.
    ///
    /// Reduces variance for monotonic functions.
    Antithetic,
}

/// Options for Monte Carlo integration.
#[derive(Debug, Clone)]
pub struct MonteCarloOptions {
    /// Number of samples (default: 10000)
    pub n_samples: usize,

    /// Sampling method (default: Plain)
    pub method: MonteCarloMethod,

    /// Random seed for reproducibility (default: None - random)
    pub seed: Option<u64>,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            method: MonteCarloMethod::Plain,
            seed: None,
        }
    }
}

impl MonteCarloOptions {
    /// Create options with specified sample count.
    pub fn with_samples(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Default::default()
        }
    }

    /// Set the sampling method.
    pub fn method(mut self, method: MonteCarloMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of Monte Carlo integration.
#[derive(Debug, Clone)]
pub struct MonteCarloResult<R: Runtime> {
    /// Computed integral value (0-D tensor)
    pub integral: Tensor<R>,
    /// Standard error estimate
    pub std_error: f64,
    /// Number of samples used
    pub n_samples: usize,
}

// ============================================================================
// Quasi-Monte Carlo Integration
// ============================================================================

/// Low-discrepancy sequence type for QMC integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QMCSequence {
    /// Sobol sequence - excellent uniformity, up to ~1000 dimensions.
    #[default]
    Sobol,

    /// Halton sequence - simpler, good for lower dimensions.
    Halton,
}

/// Options for Quasi-Monte Carlo integration.
#[derive(Debug, Clone)]
pub struct QMCOptions {
    /// Number of sample points (default: 10000)
    ///
    /// For Sobol, should ideally be a power of 2.
    pub n_samples: usize,

    /// Low-discrepancy sequence to use (default: Sobol)
    pub sequence: QMCSequence,

    /// Skip first n points (for randomization) (default: 0)
    pub skip: usize,
}

impl Default for QMCOptions {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            sequence: QMCSequence::Sobol,
            skip: 0,
        }
    }
}

impl QMCOptions {
    /// Create options with specified sample count.
    pub fn with_samples(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Default::default()
        }
    }

    /// Set the sequence type.
    pub fn sequence(mut self, sequence: QMCSequence) -> Self {
        self.sequence = sequence;
        self
    }
}

// ============================================================================
// N-dimensional Adaptive Quadrature
// ============================================================================

/// Options for n-dimensional adaptive quadrature (dblquad, tplquad, nquad).
#[derive(Debug, Clone)]
pub struct NQuadOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum subdivisions per dimension (default: 50)
    pub limit: usize,
}

impl Default for NQuadOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            limit: 50,
        }
    }
}

impl NQuadOptions {
    /// Create options with specified tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }
}

// ============================================================================
// BVP Result
// ============================================================================

/// Result of boundary value problem (BVP) solver.
#[derive(Debug, Clone)]
pub struct BVPResult<R: Runtime> {
    /// Mesh points where solution was computed (1-D tensor)
    pub x: Tensor<R>,

    /// Solution values at mesh points - shape [n_vars, n_points]
    pub y: Tensor<R>,

    /// Residual of the collocation equations (1-D tensor)
    pub residual: Tensor<R>,

    /// Whether the solver converged
    pub success: bool,

    /// Number of iterations performed
    pub niter: usize,

    /// Final mesh size (number of points)
    pub mesh_size: usize,

    /// Status message
    pub message: Option<String>,
}

impl<R: Runtime> BVPResult<R> {
    /// Get the solution at the left boundary.
    pub fn y_left(&self) -> Vec<f64> {
        let shape = self.y.shape();
        if shape.len() != 2 || shape[1] == 0 {
            return vec![];
        }
        let n_vars = shape[0];
        let all_data: Vec<f64> = self.y.to_vec();
        all_data
            .iter()
            .step_by(shape[1])
            .take(n_vars)
            .copied()
            .collect()
    }

    /// Get the solution at the right boundary.
    pub fn y_right(&self) -> Vec<f64> {
        let shape = self.y.shape();
        if shape.len() != 2 || shape[1] == 0 {
            return vec![];
        }
        let n_vars = shape[0];
        let n_points = shape[1];
        let all_data: Vec<f64> = self.y.to_vec();
        (0..n_vars)
            .map(|i| all_data[i * n_points + n_points - 1])
            .collect()
    }
}

// ============================================================================
// Symplectic Result
// ============================================================================

/// Result of symplectic integration.
///
/// For Hamiltonian systems with position q and momentum p.
#[derive(Debug, Clone)]
pub struct SymplecticResult<R: Runtime> {
    /// Time points (1-D tensor)
    pub t: Tensor<R>,

    /// Position values - shape [n_steps, n_dof]
    pub q: Tensor<R>,

    /// Momentum values - shape [n_steps, n_dof]
    pub p: Tensor<R>,

    /// Total energy at each time step (for verification)
    pub energy: Option<Tensor<R>>,

    /// Number of steps taken
    pub nsteps: usize,
}
