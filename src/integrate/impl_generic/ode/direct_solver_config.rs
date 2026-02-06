//! Configuration types for the direct sparse LU solver.
//!
//! Controls whether implicit ODE solvers (BDF, Radau, DAE) use GMRES
//! (iterative) or direct sparse LU for solving Newton systems.

/// Strategy for solving sparse linear systems in implicit ODE solvers.
///
/// # Strategies
///
/// - **Gmres**: Iterative GMRES solver with ILU preconditioning (default, backward compatible)
/// - **DirectLU**: Direct sparse LU factorization with COLAMD ordering and symbolic caching
/// - **Auto**: Chooses DirectLU for small/medium systems (n < 5000), GMRES for large systems
///
/// # When to Use DirectLU
///
/// Direct LU is preferred when:
/// - System size is moderate (n < 5000)
/// - Jacobian is highly ill-conditioned (GMRES may not converge)
/// - Repeated solves with same sparsity pattern (symbolic analysis is amortized)
/// - You need guaranteed solution accuracy (no iterative convergence issues)
#[cfg(feature = "sparse")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseSolverStrategy {
    /// Iterative GMRES solver with ILU preconditioning (default).
    #[default]
    Gmres,
    /// Direct sparse LU factorization with COLAMD ordering.
    DirectLU,
    /// Auto-select: DirectLU for n < 5000, GMRES for n >= 5000.
    Auto,
}

/// Configuration for the direct sparse LU solver.
///
/// These settings control the numeric LU factorization phase.
/// The symbolic analysis (elimination tree, reach, pattern prediction)
/// is always performed with default settings.
#[cfg(feature = "sparse")]
#[derive(Debug, Clone)]
pub struct DirectSolverConfig {
    /// Pivot tolerance for partial pivoting (default: 1.0 = full pivoting).
    ///
    /// Values in [0, 1] where 0 = no pivoting, 1 = always choose largest.
    /// Lower values reduce fill-in at the cost of numerical stability.
    pub pivot_tolerance: f64,

    /// Minimum acceptable pivot magnitude (default: 1e-12).
    ///
    /// Pivots smaller than this trigger a diagonal shift or error.
    pub pivot_threshold: f64,

    /// Diagonal shift to add when pivot is too small (default: 0.0).
    ///
    /// If > 0, adds this value to small pivots instead of failing.
    /// Useful for nearly singular matrices from ill-conditioned Jacobians.
    pub diagonal_shift: f64,

    /// Enable row/column equilibration scaling before factorization (default: false).
    ///
    /// When enabled, the matrix is scaled so that row and column norms are
    /// close to 1, which improves numerical stability for ill-conditioned
    /// Jacobians. The scaling factors are computed once during `full_analysis`
    /// and reused on subsequent solves.
    pub equilibrate: bool,

    /// Pivot growth threshold for fallback detection (default: 1e8).
    ///
    /// If the pivot growth factor from LU factorization exceeds this value,
    /// the factorization may be numerically unreliable. Callers can inspect
    /// `last_pivot_growth` to decide whether to fall back to GMRES.
    pub pivot_growth_threshold: f64,
}

#[cfg(feature = "sparse")]
impl Default for DirectSolverConfig {
    fn default() -> Self {
        Self {
            pivot_tolerance: 1.0,
            pivot_threshold: 1e-12,
            diagonal_shift: 0.0,
            equilibrate: false,
            pivot_growth_threshold: 1e8,
        }
    }
}

#[cfg(feature = "sparse")]
impl DirectSolverConfig {
    /// Create config with a diagonal shift for numerical stability.
    pub fn with_diagonal_shift(shift: f64) -> Self {
        Self {
            diagonal_shift: shift,
            ..Default::default()
        }
    }

    /// Create config with relaxed pivoting (reduces fill-in).
    pub fn relaxed_pivoting() -> Self {
        Self {
            pivot_tolerance: 0.1,
            ..Default::default()
        }
    }

    /// Create config with equilibration enabled for ill-conditioned systems.
    pub fn with_equilibration() -> Self {
        Self {
            equilibrate: true,
            ..Default::default()
        }
    }
}
