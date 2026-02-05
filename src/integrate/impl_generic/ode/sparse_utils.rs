//! Sparse matrix utilities for stiff ODE solvers.
//!
//! This module provides shared functionality for sparse Jacobian support
//! in BDF and Radau solvers.

#[cfg(feature = "sparse")]
use numr::algorithm::iterative::{GmresOptions, IterativeSolvers};
#[cfg(feature = "sparse")]
use numr::error::Result;
#[cfg(feature = "sparse")]
use numr::runtime::Runtime;
#[cfg(feature = "sparse")]
use numr::sparse::{CsrData, SparseOps};
#[cfg(feature = "sparse")]
use numr::tensor::Tensor;

#[cfg(feature = "sparse")]
use crate::integrate::ode::SparseJacobianConfig;

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
/// Currently delegates to `dense_to_csr_full`. Future optimization could
/// use the pattern to avoid scanning zero regions.
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
    // TODO: Use pattern to optimize conversion by only checking non-zero positions
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
        return Err(numr::error::Error::Internal(format!(
            "GMRES did not converge: {} iterations, residual = {}",
            result.iterations, result.residual_norm
        )));
    }

    Ok(result.solution)
}
