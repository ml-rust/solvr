//! Sparse matrix utilities for stiff ODE solvers.
//!
//! This module provides shared functionality for sparse Jacobian support
//! in BDF and Radau solvers.
//!
//! # Pattern Caching
//!
//! For repeated solves with the same sparsity pattern (e.g., Newton iterations),
//! the solver can cache the symbolic ILU factorization using `SparseJacobianCache`.
//! This provides ~10-50x speedup for the ILU precomputation.
//!
//! ```ignore
//! // First Newton iteration: compute symbolic factorization
//! let mut cache = SparseJacobianCache::new();
//! let ilu = cache.get_or_compute_ilu(&client, &jacobian_csr)?;
//!
//! // Subsequent iterations: reuse symbolic factorization
//! let ilu = cache.get_or_compute_ilu(&client, &updated_jacobian_csr)?;
//! ```

#[cfg(feature = "sparse")]
use numr::algorithm::iterative::{ConvergenceReason, GmresOptions, IterativeSolvers};
#[cfg(feature = "sparse")]
use numr::algorithm::sparse_linalg::{
    IluDecomposition, IluOptions, SparseLinAlgAlgorithms, SymbolicIlu0,
};
#[cfg(feature = "sparse")]
use numr::error::Result;
#[cfg(feature = "sparse")]
use numr::runtime::Runtime;
#[cfg(feature = "sparse")]
use numr::sparse::{CsrData, SparseOps};
#[cfg(feature = "sparse")]
use numr::tensor::Tensor;

#[cfg(feature = "sparse")]
use super::direct_solver::DirectSparseSolver;
#[cfg(feature = "sparse")]
use crate::integrate::ode::SparseJacobianConfig;

/// Cache for sparse Jacobian operations in stiff ODE solvers.
///
/// Caches the symbolic ILU factorization to avoid recomputing it
/// for each Newton iteration when the sparsity pattern is unchanged.
#[cfg(feature = "sparse")]
pub struct SparseJacobianCache {
    /// Cached symbolic ILU(0) factorization
    symbolic_ilu: Option<SymbolicIlu0>,

    /// Number of times the cache was used (for metrics)
    pub cache_hits: usize,

    /// Number of times the cache was recomputed (for metrics)
    pub cache_misses: usize,
}

#[cfg(feature = "sparse")]
impl SparseJacobianCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            symbolic_ilu: None,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get ILU decomposition, computing symbolic factorization only on first call.
    ///
    /// The symbolic factorization is cached and reused for subsequent calls
    /// with the same sparsity pattern. Only the numeric factorization is
    /// recomputed each time.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client with sparse linear algebra support
    /// * `matrix` - CSR matrix to factorize (must have same pattern as previous calls)
    /// * `options` - ILU factorization options
    ///
    /// # Returns
    ///
    /// ILU decomposition with L and U factors.
    pub fn get_or_compute_ilu<R, C>(
        &mut self,
        client: &C,
        matrix: &CsrData<R>,
        options: IluOptions,
    ) -> Result<IluDecomposition<R>>
    where
        R: Runtime,
        C: SparseLinAlgAlgorithms<R>,
    {
        // Compute symbolic factorization if not cached
        if self.symbolic_ilu.is_none() {
            self.cache_misses += 1;
            let s = client.ilu0_symbolic(matrix)?;
            self.symbolic_ilu = Some(s);
        } else {
            self.cache_hits += 1;
        }

        // Safe: we just ensured symbolic_ilu is Some above
        let symbolic = self
            .symbolic_ilu
            .as_ref()
            .expect("symbolic_ilu guaranteed to be Some after initialization above");

        // Compute numeric factorization using cached symbolic data
        client.ilu0_numeric(matrix, symbolic, options)
    }

    /// Invalidate the cache, forcing recomputation on next use.
    ///
    /// Call this when the sparsity pattern changes (e.g., adaptive mesh refinement).
    pub fn invalidate(&mut self) {
        self.symbolic_ilu = None;
    }

    /// Returns true if the cache has a symbolic factorization stored.
    pub fn has_symbolic(&self) -> bool {
        self.symbolic_ilu.is_some()
    }
}

#[cfg(feature = "sparse")]
impl Default for SparseJacobianCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a dense matrix to CSR format, dropping near-zero values.
///
/// Uses numr's `dense_to_csr` which has GPU-native implementations,
/// avoiding CPU transfers on CUDA/WebGPU backends.
///
/// # Arguments
///
/// * `client` - Runtime client with sparse operations
/// * `dense` - Dense square matrix to convert
///
/// # Returns
///
/// CSR representation with values below 1e-15 threshold dropped.
#[cfg(feature = "sparse")]
pub fn dense_to_csr_full<R, C>(client: &C, dense: &Tensor<R>) -> Result<CsrData<R>>
where
    R: Runtime,
    C: SparseOps<R>,
{
    let shape = dense.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(numr::error::Error::ShapeMismatch {
            expected: vec![shape[0], shape[0]],
            got: shape.to_vec(),
        });
    }

    client.dense_to_csr(dense, 1e-15)
}

/// Convert a dense matrix to CSR format using a known sparsity pattern.
///
/// When the sparsity pattern is known from a previous Jacobian computation,
/// this function can be used to extract values only at the non-zero positions.
///
/// # Performance Note
///
/// Currently delegates to `dense_to_csr_full` for correctness. A future
/// optimization could use the pattern's row_ptrs and col_indices to:
/// 1. Compute flat indices: flat_idx = row * ncols + col
/// 2. Gather values from the flattened dense matrix at those indices
/// 3. Create CSR with pattern structure + gathered values
///
/// This would avoid scanning the entire dense matrix and checking thresholds,
/// providing ~10-100x speedup for large sparse matrices.
#[cfg(feature = "sparse")]
pub fn dense_to_csr_with_pattern<R, C>(
    client: &C,
    dense: &Tensor<R>,
    _pattern: &CsrData<R>,
) -> Result<CsrData<R>>
where
    R: Runtime,
    C: SparseOps<R>,
{
    // For now, use full conversion which is correct but not optimized.
    // The pattern could be used to extract values only at known non-zero positions,
    // but this requires tensor gather operations that may not be universally available.
    dense_to_csr_full(client, dense)
}

/// Solve a sparse linear system using GMRES with the given configuration.
///
/// This is the shared GMRES solver logic used by both BDF and Radau solvers.
///
/// # Arguments
///
/// * `client` - Runtime client with iterative solver support
/// * `m_sparse` - CSR matrix (the iteration matrix)
/// * `b` - Right-hand side vector
/// * `sparse_config` - Sparse solver configuration (tolerance, max iterations, preconditioner)
/// * `solver_name` - Name of the calling solver for error messages ("BDF" or "Radau")
///
/// # Returns
///
/// Solution vector x such that m_sparse * x ≈ b
#[cfg(feature = "sparse")]
pub fn solve_with_gmres<R, C>(
    client: &C,
    m_sparse: &CsrData<R>,
    b: &Tensor<R>,
    sparse_config: &SparseJacobianConfig<R>,
    solver_name: &str,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: IterativeSolvers<R>,
{
    let gmres_opts = GmresOptions {
        max_iter: sparse_config.max_gmres_iter,
        rtol: sparse_config.gmres_tol,
        atol: 1e-14,
        preconditioner: sparse_config.preconditioner,
        ..Default::default()
    };

    let result = client.gmres(m_sparse, b, None, gmres_opts).map_err(|e| {
        numr::error::Error::Internal(format!(
            "GMRES failed in sparse {} solve: {}",
            solver_name, e
        ))
    })?;

    if !result.converged {
        // Provide actionable error messages based on convergence reason
        let diag = &result.diagnostics;
        let hint = match result.reason {
            ConvergenceReason::MaxIterationsReached => format!(
                "Consider: (1) Increase max_gmres_iter (currently {}), \
                 (2) Enable ILU preconditioning if not already, \
                 (3) Loosen gmres_tol (currently {:.2e})",
                diag.max_iter, diag.rtol
            ),
            ConvergenceReason::Stagnation => format!(
                "Residual stagnated at {:.2e} (initial: {:.2e}). \
                 System may be ill-conditioned. Consider: \
                 (1) Enable ILU preconditioning, \
                 (2) Increase restart parameter, \
                 (3) Use a smaller time step",
                result.residual_norm, diag.initial_residual_norm
            ),
            ConvergenceReason::NumericalBreakdown => format!(
                "Numerical breakdown at iteration {}. Matrix may be singular. \
                 Check: (1) Jacobian computation is correct, \
                 (2) Problem is well-posed, \
                 (3) Consider adding a diagonal shift",
                result.iterations
            ),
            _ => format!(
                "Unexpected non-convergence reason: {}. {}",
                result.reason,
                result.reason.hint()
            ),
        };

        return Err(numr::error::Error::Internal(format!(
            "GMRES did not converge in {} {}: {} iterations, residual = {:.2e}. {}",
            solver_name, "Newton step", result.iterations, result.residual_norm, hint
        )));
    }

    Ok(result.solution)
}

/// Solve a sparse linear system using direct sparse LU factorization.
///
/// Uses the `DirectSparseSolver` to convert the dense iteration matrix to
/// sparse format, apply COLAMD ordering, compute symbolic/numeric LU
/// factorization, and solve via forward/backward substitution.
///
/// # Arguments
///
/// * `client` - Runtime client with sparse operations
/// * `direct_solver` - Mutable reference to the direct solver (caches symbolic analysis)
/// * `m_dense` - Dense iteration matrix
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// Solution vector x such that m_dense * x = b
#[cfg(feature = "sparse")]
pub fn solve_with_direct_lu<R, C>(
    client: &C,
    direct_solver: &mut DirectSparseSolver<R>,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SparseOps<R> + numr::ops::IndexingOps<R> + numr::ops::TensorOps<R> + numr::ops::ScalarOps<R>,
{
    direct_solver.solve(client, m_dense, b)
}

/// Unified sparse system solver for BDF/Radau/DAE solvers.
///
/// Dispatches to direct LU (if available) or falls back to GMRES.
/// The dense path is NOT included here — each solver handles dense separately
/// since Radau requires reshape operations.
///
/// # Arguments
///
/// * `client` - Runtime client with sparse operations
/// * `m_dense` - Dense iteration matrix
/// * `b` - Right-hand side vector
/// * `sparse_config` - Sparse Jacobian configuration
/// * `direct_solver` - Optional direct LU solver (may be None for Auto/GMRES strategy)
/// * `pattern` - Optional CSR pattern for optimized dense→CSR conversion
/// * `solver_name` - Name for error messages ("BDF", "Radau", "DAE")
///
/// # Returns
///
/// Solution vector x such that m_dense * x = b
#[cfg(feature = "sparse")]
pub fn solve_sparse_system<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    sparse_config: &crate::integrate::ode::SparseJacobianConfig<R>,
    direct_solver: &mut Option<DirectSparseSolver<R>>,
    pattern: Option<&CsrData<R>>,
    solver_name: &str,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SparseOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::ScalarOps<R>
        + IterativeSolvers<R>,
{
    // Use direct LU if a solver is available
    if let Some(ds) = direct_solver.as_mut() {
        return solve_with_direct_lu(client, ds, m_dense, b);
    }

    // Fall back to GMRES
    let m_sparse = if let Some(pat) = pattern {
        dense_to_csr_with_pattern(client, m_dense, pat)?
    } else {
        dense_to_csr_full(client, m_dense)?
    };

    solve_with_gmres(client, &m_sparse, b, sparse_config, solver_name)
}

/// Create a DirectSparseSolver based on configuration and problem size.
///
/// # Arguments
///
/// * `sparse_config` - Sparse Jacobian configuration
/// * `n` - Problem size (state dimension)
///
/// # Returns
///
/// Some(DirectSparseSolver) if strategy is DirectLU or Auto (n < 5000), None otherwise
#[cfg(feature = "sparse")]
pub fn create_direct_solver<R: Runtime>(
    sparse_config: &crate::integrate::ode::SparseJacobianConfig<R>,
    n: usize,
) -> Option<DirectSparseSolver<R>> {
    use super::direct_solver_config::SparseSolverStrategy;

    if !sparse_config.enabled {
        return None;
    }

    match sparse_config.solver_strategy {
        SparseSolverStrategy::DirectLU => {
            Some(DirectSparseSolver::new(&sparse_config.direct_solver_config))
        }
        SparseSolverStrategy::Auto if n < 5000 => {
            Some(DirectSparseSolver::new(&sparse_config.direct_solver_config))
        }
        _ => None,
    }
}
