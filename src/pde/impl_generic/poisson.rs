//! Poisson equation solver: -nabla^2 u = f on 2D grid.
//!
//! Supports Dirichlet, Neumann (ghost-node), and Periodic boundary conditions.
use crate::DType;

use numr::algorithm::iterative::IterativeSolvers;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundarySpec, FdmOptions, FdmResult, Grid2D, SparseSolver};

use super::boundary::{extract_boundary_values_2d, side_conditions_2d};
use super::solve_sparse::solve_sparse_system;
use super::stencil::{
    SideBc, assemble_neg_laplacian_2d_dirichlet, assemble_neg_laplacian_2d_mixed,
};

/// Solve the Poisson equation -nabla^2 u = f on a 2D grid.
///
/// Dispatches to the appropriate stencil assembler based on the boundary
/// conditions supplied:
///
/// - **Dirichlet**: Standard 5-point stencil; boundary rows become identity
///   rows and boundary contributions shift to the RHS. The resulting system
///   is symmetric positive definite; CG is the default solver.
///
/// - **Neumann** (`du/dn = g`): Ghost-node discretisation. For each boundary
///   side the stencil is modified so the boundary row carries coefficient
///   `2/h²` toward its one interior neighbour and `2/h²` on the diagonal, with
///   a flux term `2g/h` added to the RHS. The resulting system is singular
///   (solution is determined only up to an additive constant). We fix the gauge
///   by setting `u[0,0] = 0`: the first row/column is replaced with a unit row
///   and the RHS is set to zero there. GMRES is used (the gauge-fixed system is
///   non-SPD). The returned solution satisfies `u[0,0] = 0`.
///
/// - **Periodic**: Wrapping 5-point stencil where every node's left/right/bottom/top
///   neighbours wrap around the grid. The resulting system is singular for the same
///   reason as Neumann; the gauge fix (`u[0,0] = 0`) is applied identically.
///   GMRES is used.
pub fn poisson_impl<R, C>(
    client: &C,
    f: &Tensor<R>,
    grid: &Grid2D,
    boundary: &[BoundarySpec<R>],
    options: &FdmOptions,
) -> PdeResult<FdmResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + IterativeSolvers<R> + RuntimeClient<R>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let device = client.device();

    if nx < 3 || ny < 3 {
        return Err(PdeError::InvalidGrid {
            context: format!("Grid must be at least 3x3, got {}x{}", nx, ny),
        });
    }

    // Per-side BC kinds (validates Periodic pairing). Dirichlet boundary values
    // are read by the assembler only for nodes lying on a Dirichlet side.
    let sides = side_conditions_2d(boundary)?;
    let dirichlet_values = extract_boundary_values_2d(boundary, nx, ny)?;

    // Load f into rhs_data (interior nodes); boundary rows are overwritten below.
    let f_data: Vec<f64> = f.to_vec();
    let mut rhs_data = vec![0.0f64; n];
    rhs_data[..n.min(f_data.len())].copy_from_slice(&f_data[..n.min(f_data.len())]);

    // Pure Dirichlet → symmetric positive-definite system; use the dedicated SPD
    // assembler (and CG by default).
    if sides.iter().all(|s| *s == SideBc::Dirichlet) {
        let a = assemble_neg_laplacian_2d_dirichlet::<R>(
            grid,
            &dirichlet_values,
            &mut rhs_data,
            device,
        )
        .map_err(PdeError::from)?;

        let rhs = Tensor::<R>::from_slice(&rhs_data, &[n], device);
        let result = solve_sparse_system(client, &a, &rhs, options, "Poisson solver (Dirichlet)")?;

        return Ok(FdmResult {
            solution: result.solution.reshape(&[nx, ny])?,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
        });
    }

    // General path: any combination of Neumann / Periodic / Dirichlet sides.
    let a_raw = assemble_neg_laplacian_2d_mixed::<R>(
        grid,
        &sides,
        &dirichlet_values,
        &mut rhs_data,
        device,
    )
    .map_err(PdeError::from)?;

    // A system with at least one Dirichlet side is non-singular. With no
    // Dirichlet side (all Neumann/Periodic) the solution is defined only up to
    // an additive constant, so we fix the gauge by pinning u[0,0] = 0.
    let has_dirichlet = sides.contains(&SideBc::Dirichlet);
    let (a, label) = if has_dirichlet {
        (a_raw, "Poisson solver (mixed)")
    } else {
        let a = pin_first_dof(a_raw, n, device)?;
        rhs_data[0] = 0.0;
        (a, "Poisson solver (Neumann/Periodic)")
    };

    let rhs = Tensor::<R>::from_slice(&rhs_data, &[n], device);

    // The ghost-node Neumann rows (and the gauge fix) make the system
    // non-symmetric; use GMRES unless the caller explicitly picked a solver.
    let opts = options_with_gmres_fallback(options);
    let result = solve_sparse_system(client, &a, &rhs, &opts, label)?;

    Ok(FdmResult {
        solution: result.solution.reshape(&[nx, ny])?,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Replace row 0 of `a` with an identity row: A[0,0]=1, all other A[0,*]=0.
///
/// Used to fix the gauge for Neumann and Periodic systems that would otherwise
/// be singular. The CSR row_ptrs, col_indices, and values are rebuilt host-side.
fn pin_first_dof<R: Runtime<DType = DType>>(
    a: numr::sparse::CsrData<R>,
    n: usize,
    device: &R::Device,
) -> PdeResult<numr::sparse::CsrData<R>> {
    use numr::sparse::CsrData;

    let row_ptrs_t = a.row_ptrs();
    let col_idx_t = a.col_indices();
    let values_t = a.values();

    let row_ptrs_h: Vec<i64> = row_ptrs_t.to_vec();
    let col_idx_h: Vec<i64> = col_idx_t.to_vec();
    let values_h: Vec<f64> = values_t.to_vec();

    // Number of non-zeros in row 0 in the original matrix.
    let row0_start = row_ptrs_h[0] as usize;
    let row0_end = row_ptrs_h[1] as usize;
    let orig_nnz_row0 = row0_end - row0_start;

    // We replace row 0 with a single entry (0,0)=1.
    // Delta in nnz: 1 - orig_nnz_row0
    let delta: i64 = 1 - orig_nnz_row0 as i64;
    let orig_total_nnz = col_idx_h.len();
    let new_total_nnz = (orig_total_nnz as i64 + delta) as usize;

    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_cols = vec![0i64; new_total_nnz];
    let mut new_vals = vec![0.0f64; new_total_nnz];

    // Row 0: single entry (0, 0) = 1.0
    new_row_ptrs[0] = 0;
    new_row_ptrs[1] = 1;
    new_cols[0] = 0;
    new_vals[0] = 1.0;

    // Remaining rows: copy, adjusting offsets by delta.
    let mut dst = 1usize; // write cursor in new_cols/new_vals
    for row in 1..n {
        let src_start = row_ptrs_h[row] as usize;
        let src_end = row_ptrs_h[row + 1] as usize;
        let len = src_end - src_start;
        new_cols[dst..dst + len].copy_from_slice(&col_idx_h[src_start..src_end]);
        new_vals[dst..dst + len].copy_from_slice(&values_h[src_start..src_end]);
        dst += len;
        new_row_ptrs[row + 1] = dst as i64;
    }

    let rp = Tensor::<R>::from_slice(&new_row_ptrs, &[n + 1], device);
    let ci = Tensor::<R>::from_slice(&new_cols, &[new_total_nnz], device);
    let vv = Tensor::<R>::from_slice(&new_vals, &[new_total_nnz], device);

    CsrData::new(rp, ci, vv, [n, n]).map_err(PdeError::from)
}

/// Return options that use GMRES if the caller did not explicitly pick one.
///
/// Neumann and Periodic gauge-fixed systems are non-symmetric, so CG (the
/// default) would not converge. We fall back to GMRES while preserving all
/// other parameters the caller specified.
fn options_with_gmres_fallback(options: &FdmOptions) -> FdmOptions {
    if options.solver == SparseSolver::Cg {
        FdmOptions {
            solver: SparseSolver::Gmres,
            ..options.clone()
        }
    } else {
        options.clone()
    }
}
