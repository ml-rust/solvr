//! Unified client trait for stiff ODE solvers (BDF, Radau).
//!
//! This module provides a single `StiffSolverClient` trait that conditionally
//! includes `IterativeSolvers` and `SparseOps` when the sparse feature is enabled,
//! avoiding trait duplication between solvers.

use numr::ops::{LinalgOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};

#[cfg(feature = "sparse")]
use numr::algorithm::iterative::IterativeSolvers;
#[cfg(feature = "sparse")]
use numr::sparse::SparseOps;

/// Client trait for stiff ODE solvers requiring linear algebra operations.
///
/// When the `sparse` feature is enabled, this trait also requires `IterativeSolvers`
/// for GMRES-based sparse linear system solving and `SparseOps` for dense-to-CSR
/// conversion.
#[cfg(feature = "sparse")]
pub trait StiffSolverClient<R: Runtime>:
    TensorOps<R>
    + ScalarOps<R>
    + LinalgOps<R>
    + UtilityOps<R>
    + RuntimeClient<R>
    + IterativeSolvers<R>
    + SparseOps<R>
{
}

#[cfg(feature = "sparse")]
impl<R, T> StiffSolverClient<R> for T
where
    R: Runtime,
    T: TensorOps<R>
        + ScalarOps<R>
        + LinalgOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>
        + IterativeSolvers<R>
        + SparseOps<R>,
{
}

/// Client trait for stiff ODE solvers requiring linear algebra operations.
///
/// Without the `sparse` feature, only dense linear algebra is available.
#[cfg(not(feature = "sparse"))]
pub trait StiffSolverClient<R: Runtime>:
    TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>
{
}

#[cfg(not(feature = "sparse"))]
impl<R, T> StiffSolverClient<R> for T
where
    R: Runtime,
    T: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
}
