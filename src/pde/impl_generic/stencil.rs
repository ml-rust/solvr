//! Sparse stencil assembly for finite difference methods.
//!
//! Builds sparse Laplacian matrices from grid specifications.
//! Stencil assembly is a one-time setup cost at API boundary.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::CsrData;
use numr::tensor::Tensor;

use crate::pde::types::{Grid2D, Grid3D};

/// Assemble a 2D negative Laplacian (-nabla^2) with Dirichlet BCs as CSR.
///
/// Boundary rows become identity rows. Interior neighbors that are boundary
/// nodes have their contribution moved to the RHS vector.
///
/// This solves: -nabla^2 u = f, so the matrix is SPD for interior nodes.
pub fn assemble_neg_laplacian_2d_dirichlet<R>(
    grid: &Grid2D,
    boundary_values: &[f64],
    rhs_data: &mut [f64],
    device: &R::Device,
) -> Result<CsrData<R>>
where
    R: Runtime<DType = DType>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let is_boundary =
        |i: usize, j: usize| -> bool { i == 0 || i == nx - 1 || j == 0 || j == ny - 1 };

    let bval = |idx: usize| -> f64 {
        if idx < boundary_values.len() {
            boundary_values[idx]
        } else {
            0.0
        }
    };

    let mut rows = Vec::with_capacity(5 * n);
    let mut cols = Vec::with_capacity(5 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(5 * n);

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;

            if is_boundary(i, j) {
                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(1.0);
                rhs_data[idx] = bval(idx);
            } else {
                let mut center = 0.0;

                // Left (i-1, j)
                let left_idx = (i - 1) * ny + j;
                if is_boundary(i - 1, j) {
                    rhs_data[idx] += bval(left_idx) / dx2;
                } else {
                    rows.push(idx as i64);
                    cols.push(left_idx as i64);
                    vals.push(-1.0 / dx2);
                }
                center += 1.0 / dx2;

                // Right (i+1, j)
                let right_idx = (i + 1) * ny + j;
                if is_boundary(i + 1, j) {
                    rhs_data[idx] += bval(right_idx) / dx2;
                } else {
                    rows.push(idx as i64);
                    cols.push(right_idx as i64);
                    vals.push(-1.0 / dx2);
                }
                center += 1.0 / dx2;

                // Bottom (i, j-1)
                let bottom_idx = i * ny + (j - 1);
                if is_boundary(i, j - 1) {
                    rhs_data[idx] += bval(bottom_idx) / dy2;
                } else {
                    rows.push(idx as i64);
                    cols.push(bottom_idx as i64);
                    vals.push(-1.0 / dy2);
                }
                center += 1.0 / dy2;

                // Top (i, j+1)
                let top_idx = i * ny + (j + 1);
                if is_boundary(i, j + 1) {
                    rhs_data[idx] += bval(top_idx) / dy2;
                } else {
                    rows.push(idx as i64);
                    cols.push(top_idx as i64);
                    vals.push(-1.0 / dy2);
                }
                center += 1.0 / dy2;

                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(center);
            }
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    // COO to CSR: sort by row, build row_ptrs
    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

/// Assemble 2D Laplacian (not negated) as CSR for time-dependent solvers.
///
/// This returns L such that L*u approximates nabla^2 u.
/// Used by heat/wave equation solvers where the sign is handled externally.
pub fn assemble_laplacian_2d<R>(grid: &Grid2D, device: &R::Device) -> Result<CsrData<R>>
where
    R: Runtime<DType = DType>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let mut rows = Vec::with_capacity(5 * n);
    let mut cols = Vec::with_capacity(5 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(5 * n);

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;
            let mut center = 0.0;

            if i > 0 {
                rows.push(idx as i64);
                cols.push((idx - ny) as i64);
                vals.push(1.0 / dx2);
                center -= 1.0 / dx2;
            }
            if i < nx - 1 {
                rows.push(idx as i64);
                cols.push((idx + ny) as i64);
                vals.push(1.0 / dx2);
                center -= 1.0 / dx2;
            }
            if j > 0 {
                rows.push(idx as i64);
                cols.push((idx - 1) as i64);
                vals.push(1.0 / dy2);
                center -= 1.0 / dy2;
            }
            if j < ny - 1 {
                rows.push(idx as i64);
                cols.push((idx + 1) as i64);
                vals.push(1.0 / dy2);
                center -= 1.0 / dy2;
            }

            rows.push(idx as i64);
            cols.push(idx as i64);
            vals.push(center);
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

/// Assemble 3D Laplacian as CSR (7-point stencil).
pub fn assemble_laplacian_3d<R>(grid: &Grid3D, device: &R::Device) -> Result<CsrData<R>>
where
    R: Runtime<DType = DType>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let n = nx * ny * nz;
    let nyz = ny * nz;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;
    let dz2 = grid.dz * grid.dz;

    let mut rows = Vec::with_capacity(7 * n);
    let mut cols = Vec::with_capacity(7 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(7 * n);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = i * nyz + j * nz + k;
                let mut center = 0.0;

                if i > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - nyz) as i64);
                    vals.push(1.0 / dx2);
                    center -= 1.0 / dx2;
                }
                if i < nx - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + nyz) as i64);
                    vals.push(1.0 / dx2);
                    center -= 1.0 / dx2;
                }
                if j > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - nz) as i64);
                    vals.push(1.0 / dy2);
                    center -= 1.0 / dy2;
                }
                if j < ny - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + nz) as i64);
                    vals.push(1.0 / dy2);
                    center -= 1.0 / dy2;
                }
                if k > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - 1) as i64);
                    vals.push(1.0 / dz2);
                    center -= 1.0 / dz2;
                }
                if k < nz - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + 1) as i64);
                    vals.push(1.0 / dz2);
                    center -= 1.0 / dz2;
                }

                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(center);
            }
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

// ============================================================================
// General (per-side) BC stencil
// ============================================================================

/// Per-side boundary condition for the general Poisson assembler.
///
/// Used by [`assemble_neg_laplacian_2d_mixed`] to express any combination of
/// Dirichlet, Neumann, and Periodic conditions across the four grid sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SideBc {
    /// Fixed value (supplied per node via `dirichlet_values`).
    Dirichlet,
    /// Fixed outward normal derivative `du/dn = g`.
    Neumann(f64),
    /// Periodic wrap-around (must be paired with the opposite side).
    Periodic,
}

/// Assemble a 2D negative Laplacian (-nabla^2) with arbitrary per-side BCs.
///
/// This is the single general assembler underpinning every non-pure-Dirichlet
/// Poisson configuration: uniform Neumann, uniform Periodic, and any
/// heterogeneous (mixed) per-side combination are all handled here.
///
/// `sides` = `[left, right, bottom, top]`. `dirichlet_values` is a flat
/// `[nx*ny]` row-major array giving the fixed value at each node lying on a
/// Dirichlet side (entries for non-Dirichlet nodes are ignored).
///
/// Per-node discretisation of `(i, j)`:
/// - **Dirichlet dominance**: a node on any Dirichlet side becomes an identity
///   row `u = dirichlet_values[idx]`. A fixed value overrides flux/periodicity,
///   so corners shared between a Dirichlet side and another side resolve
///   deterministically to Dirichlet.
/// - **Neumann** side (`du/dn = g`, second-order ghost-node elimination): the
///   boundary row gets `+2/h^2` on the diagonal and `-2/h^2` toward its single
///   interior neighbour, with `+2g/h` added to the RHS.
/// - **Periodic** side: the missing neighbour wraps to the opposite edge with
///   the standard `-1/h^2` coefficient.
/// - A neighbour that is itself a Dirichlet node contributes its known value to
///   the RHS instead of producing a matrix entry.
///
/// Periodic sides MUST be paired (left&right, or bottom&top); the caller is
/// responsible for validating this.
///
/// A configuration with no Dirichlet side (all Neumann/Periodic) yields a
/// singular system (solution defined up to an additive constant); the caller
/// must fix the gauge before solving.
pub fn assemble_neg_laplacian_2d_mixed<R>(
    grid: &Grid2D,
    sides: &[SideBc; 4],
    dirichlet_values: &[f64],
    rhs_data: &mut [f64],
    device: &R::Device,
) -> Result<CsrData<R>>
where
    R: Runtime<DType = DType>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let dx = grid.dx;
    let dy = grid.dy;
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let [s_left, s_right, s_bottom, s_top] = *sides;

    // A node is a fixed-value (Dirichlet) node iff it lies on any Dirichlet side.
    let is_dir = |i: usize, j: usize| -> bool {
        (i == 0 && s_left == SideBc::Dirichlet)
            || (i == nx - 1 && s_right == SideBc::Dirichlet)
            || (j == 0 && s_bottom == SideBc::Dirichlet)
            || (j == ny - 1 && s_top == SideBc::Dirichlet)
    };
    let dval = |idx: usize| -> f64 { dirichlet_values.get(idx).copied().unwrap_or(0.0) };

    let mut rows: Vec<i64> = Vec::with_capacity(5 * n);
    let mut cols: Vec<i64> = Vec::with_capacity(5 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(5 * n);

    // Add a standard (coefficient -1/h^2) neighbour at grid position `(ni, nj)`.
    // If that neighbour is a Dirichlet node its known value is folded into the
    // RHS; otherwise a matrix entry is emitted. The centre coefficient grows by
    // 1/h^2 either way.
    #[allow(clippy::too_many_arguments)]
    fn add_neighbour(
        ni: usize,
        nj: usize,
        ny: usize,
        inv_h2: f64,
        idx: usize,
        is_dir: &impl Fn(usize, usize) -> bool,
        dval: &impl Fn(usize) -> f64,
        rows: &mut Vec<i64>,
        cols: &mut Vec<i64>,
        vals: &mut Vec<f64>,
        rhs_data: &mut [f64],
        center: &mut f64,
    ) {
        let nb = ni * ny + nj;
        if is_dir(ni, nj) {
            rhs_data[idx] += dval(nb) * inv_h2;
        } else {
            rows.push(idx as i64);
            cols.push(nb as i64);
            vals.push(-inv_h2);
        }
        *center += inv_h2;
    }

    let inv_dx2 = 1.0 / dx2;
    let inv_dy2 = 1.0 / dy2;

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;

            // Dirichlet dominance: identity row u = dirichlet_values[idx].
            if is_dir(i, j) {
                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(1.0);
                rhs_data[idx] = dval(idx);
                continue;
            }

            let on_left = i == 0;
            let on_right = i == nx - 1;
            let on_bottom = j == 0;
            let on_top = j == ny - 1;
            let mut center = 0.0f64;

            // ---- x-axis ----
            if on_left {
                if let SideBc::Neumann(g) = s_left {
                    // Ghost-node fully accounts for the x-direction at this node.
                    center += 2.0 / dx2;
                    let nb = (i + 1) * ny + j;
                    if is_dir(i + 1, j) {
                        rhs_data[idx] += (2.0 / dx2) * dval(nb);
                    } else {
                        rows.push(idx as i64);
                        cols.push(nb as i64);
                        vals.push(-2.0 / dx2);
                    }
                    rhs_data[idx] += 2.0 * g / dx;
                } else {
                    // Periodic left edge: wrap to i = nx-1, plus the normal right neighbour.
                    add_neighbour(
                        nx - 1,
                        j,
                        ny,
                        inv_dx2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                    add_neighbour(
                        i + 1,
                        j,
                        ny,
                        inv_dx2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                }
            } else if on_right {
                if let SideBc::Neumann(g) = s_right {
                    center += 2.0 / dx2;
                    let nb = (i - 1) * ny + j;
                    if is_dir(i - 1, j) {
                        rhs_data[idx] += (2.0 / dx2) * dval(nb);
                    } else {
                        rows.push(idx as i64);
                        cols.push(nb as i64);
                        vals.push(-2.0 / dx2);
                    }
                    rhs_data[idx] += 2.0 * g / dx;
                } else {
                    // Periodic right edge: normal left neighbour, plus wrap to i = 0.
                    add_neighbour(
                        i - 1,
                        j,
                        ny,
                        inv_dx2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                    add_neighbour(
                        0,
                        j,
                        ny,
                        inv_dx2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                }
            } else {
                // Interior in x: both neighbours.
                add_neighbour(
                    i - 1,
                    j,
                    ny,
                    inv_dx2,
                    idx,
                    &is_dir,
                    &dval,
                    &mut rows,
                    &mut cols,
                    &mut vals,
                    rhs_data,
                    &mut center,
                );
                add_neighbour(
                    i + 1,
                    j,
                    ny,
                    inv_dx2,
                    idx,
                    &is_dir,
                    &dval,
                    &mut rows,
                    &mut cols,
                    &mut vals,
                    rhs_data,
                    &mut center,
                );
            }

            // ---- y-axis ----
            if on_bottom {
                if let SideBc::Neumann(g) = s_bottom {
                    center += 2.0 / dy2;
                    let nb = i * ny + (j + 1);
                    if is_dir(i, j + 1) {
                        rhs_data[idx] += (2.0 / dy2) * dval(nb);
                    } else {
                        rows.push(idx as i64);
                        cols.push(nb as i64);
                        vals.push(-2.0 / dy2);
                    }
                    rhs_data[idx] += 2.0 * g / dy;
                } else {
                    add_neighbour(
                        i,
                        ny - 1,
                        ny,
                        inv_dy2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                    add_neighbour(
                        i,
                        j + 1,
                        ny,
                        inv_dy2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                }
            } else if on_top {
                if let SideBc::Neumann(g) = s_top {
                    center += 2.0 / dy2;
                    let nb = i * ny + (j - 1);
                    if is_dir(i, j - 1) {
                        rhs_data[idx] += (2.0 / dy2) * dval(nb);
                    } else {
                        rows.push(idx as i64);
                        cols.push(nb as i64);
                        vals.push(-2.0 / dy2);
                    }
                    rhs_data[idx] += 2.0 * g / dy;
                } else {
                    add_neighbour(
                        i,
                        j - 1,
                        ny,
                        inv_dy2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                    add_neighbour(
                        i,
                        0,
                        ny,
                        inv_dy2,
                        idx,
                        &is_dir,
                        &dval,
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        rhs_data,
                        &mut center,
                    );
                }
            } else {
                add_neighbour(
                    i,
                    j - 1,
                    ny,
                    inv_dy2,
                    idx,
                    &is_dir,
                    &dval,
                    &mut rows,
                    &mut cols,
                    &mut vals,
                    rhs_data,
                    &mut center,
                );
                add_neighbour(
                    i,
                    j + 1,
                    ny,
                    inv_dy2,
                    idx,
                    &is_dir,
                    &dval,
                    &mut rows,
                    &mut cols,
                    &mut vals,
                    rhs_data,
                    &mut center,
                );
            }

            rows.push(idx as i64);
            cols.push(idx as i64);
            vals.push(center);
        }
    }

    coo_to_csr_unsorted::<R>(&rows, &cols, &vals, [n, n], device)
}

/// Convert unsorted COO data to CSR format (sorts by row then column).
fn coo_to_csr_unsorted<R: Runtime<DType = DType>>(
    row_vec: &[i64],
    col_vec: &[i64],
    val_vec: &[f64],
    shape: [usize; 2],
    device: &R::Device,
) -> Result<CsrData<R>> {
    let [nrows, _ncols] = shape;
    let nnz = row_vec.len();

    // Sort by (row, col)
    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&k| (row_vec[k], col_vec[k]));

    let mut sorted_rows = vec![0i64; nnz];
    let mut sorted_cols = vec![0i64; nnz];
    let mut sorted_vals = vec![0.0f64; nnz];
    for (dst, &src) in order.iter().enumerate() {
        sorted_rows[dst] = row_vec[src];
        sorted_cols[dst] = col_vec[src];
        sorted_vals[dst] = val_vec[src];
    }

    let mut row_ptrs = vec![0i64; nrows + 1];
    for &r in &sorted_rows {
        row_ptrs[r as usize + 1] += 1;
    }
    for i in 1..=nrows {
        row_ptrs[i] += row_ptrs[i - 1];
    }
    debug_assert_eq!(row_ptrs[nrows] as usize, nnz);

    let rp_tensor = Tensor::<R>::from_slice(&row_ptrs, &[nrows + 1], device);
    let col_tensor = Tensor::<R>::from_slice(&sorted_cols, &[nnz], device);
    let val_tensor = Tensor::<R>::from_slice(&sorted_vals, &[nnz], device);

    CsrData::new(rp_tensor, col_tensor, val_tensor, shape)
}

/// Convert already-sorted COO data to CSR format.
///
/// The stencil assembly produces entries sorted by row (and within each row by column),
/// so we can build row_ptrs directly without a general sort.
fn coo_to_csr_sorted<R: Runtime<DType = DType>>(
    row_indices: &Tensor<R>,
    col_indices: &Tensor<R>,
    values: &Tensor<R>,
    shape: [usize; 2],
    device: &R::Device,
) -> Result<CsrData<R>> {
    let [nrows, _ncols] = shape;
    let rows_vec: Vec<i64> = row_indices.to_vec();
    let nnz = rows_vec.len();

    // Build row_ptrs
    let mut row_ptrs = vec![0i64; nrows + 1];
    for &r in &rows_vec {
        row_ptrs[r as usize + 1] += 1;
    }
    // Cumulative sum
    for i in 1..=nrows {
        row_ptrs[i] += row_ptrs[i - 1];
    }
    debug_assert_eq!(row_ptrs[nrows] as usize, nnz);

    let rp_tensor = Tensor::<R>::from_slice(&row_ptrs, &[nrows + 1], device);
    CsrData::new(rp_tensor, col_indices.clone(), values.clone(), shape)
}
