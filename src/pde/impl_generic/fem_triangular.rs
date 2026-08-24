//! 2D triangular FEM: -nabla^2 u = f on a triangular mesh.
//!
//! Uses linear triangular basis functions with sparse global assembly.
//! Element stiffness matrices are computed from vertex coordinates.
use crate::DType;

use numr::algorithm::iterative::IterativeSolvers;
use numr::error::Result;
use numr::ops::{LinalgOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::CsrData;
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{FdmOptions, FemResult};

use super::solve_sparse::solve_sparse_system;

/// Solve -nabla^2 u = f on a 2D triangular mesh.
///
/// # Arguments
/// * `f_rhs` - Right-hand side evaluated at nodes, shape `[n_nodes]`
/// * `nodes` - Node coordinates, shape `[n_nodes, 2]`
/// * `elements` - Triangle connectivity, shape `[n_elements, 3]` (I64 indices)
/// * `boundary_nodes` - Indices of boundary nodes, shape `[n_boundary]` (I64)
/// * `boundary_values` - Dirichlet values at boundary nodes, shape `[n_boundary]`
pub fn fem_triangular_impl<R, C>(
    client: &C,
    f_rhs: &Tensor<R>,
    nodes: &Tensor<R>,
    elements: &Tensor<R>,
    boundary_nodes: &Tensor<R>,
    boundary_values: &Tensor<R>,
    options: &FdmOptions,
) -> PdeResult<FemResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + IterativeSolvers<R> + RuntimeClient<R>,
{
    let node_shape = nodes.shape();
    let elem_shape = elements.shape();
    let device = client.device();

    let n_nodes = node_shape[0];
    let n_elements = elem_shape[0];

    if node_shape.len() != 2 || node_shape[1] != 2 {
        return Err(PdeError::InvalidParameter {
            parameter: "nodes".to_string(),
            message: format!("Expected shape [n, 2], got {:?}", node_shape),
        });
    }
    if elem_shape.len() != 2 || elem_shape[1] != 3 {
        return Err(PdeError::InvalidParameter {
            parameter: "elements".to_string(),
            message: format!("Expected shape [n, 3], got {:?}", elem_shape),
        });
    }

    // Extract data to CPU for assembly (API boundary - one-time cost)
    let node_data: Vec<f64> = nodes.to_vec();
    let elem_data: Vec<i64> = elements.to_vec();
    let f_data: Vec<f64> = f_rhs.to_vec();
    let bc_node_data: Vec<i64> = boundary_nodes.to_vec();
    let bc_val_data: Vec<f64> = boundary_values.to_vec();

    // Build global stiffness matrix in COO format and load vector
    let mut rows = Vec::with_capacity(9 * n_elements);
    let mut cols = Vec::with_capacity(9 * n_elements);
    let mut vals: Vec<f64> = Vec::with_capacity(9 * n_elements);
    let mut load = vec![0.0; n_nodes];

    for e in 0..n_elements {
        let i0 = elem_data[e * 3] as usize;
        let i1 = elem_data[e * 3 + 1] as usize;
        let i2 = elem_data[e * 3 + 2] as usize;

        // Vertex coordinates
        let x0 = node_data[i0 * 2];
        let y0 = node_data[i0 * 2 + 1];
        let x1 = node_data[i1 * 2];
        let y1 = node_data[i1 * 2 + 1];
        let x2 = node_data[i2 * 2];
        let y2 = node_data[i2 * 2 + 1];

        // Signed area
        let area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = area2.abs() / 2.0;

        if area < 1e-15 {
            return Err(PdeError::SingularSystem {
                context: format!("Degenerate triangle at element {}", e),
            });
        }

        // Gradient of basis functions (constant on linear triangles):
        // grad(phi_i) = [y_j - y_k, x_k - x_j] / (2*area)
        let grad = [
            [(y1 - y2) / area2, (x2 - x1) / area2],
            [(y2 - y0) / area2, (x0 - x2) / area2],
            [(y0 - y1) / area2, (x1 - x0) / area2],
        ];

        // Element stiffness: K_e[i][j] = area * dot(grad_i, grad_j)
        let nodes_e = [i0, i1, i2];
        for li in 0..3 {
            for lj in 0..3 {
                let dot = grad[li][0] * grad[lj][0] + grad[li][1] * grad[lj][1];
                let k_val = area * dot;
                rows.push(nodes_e[li] as i64);
                cols.push(nodes_e[lj] as i64);
                vals.push(k_val);
            }
        }

        // Lumped load: f_e[i] = (area / 3) * f(node_i)
        let load_factor = area / 3.0;
        for &ni in &nodes_e {
            load[ni] += load_factor * f_data[ni];
        }
    }

    // Apply Dirichlet BCs: set boundary rows to identity and adjust RHS
    let mut bc_set = vec![false; n_nodes];
    let mut bc_vals_map = vec![0.0; n_nodes];
    for (idx, &node_idx) in bc_node_data.iter().enumerate() {
        let ni = node_idx as usize;
        bc_set[ni] = true;
        bc_vals_map[ni] = if idx < bc_val_data.len() {
            bc_val_data[idx]
        } else {
            0.0
        };
    }

    // Filter COO entries: boundary rows become identity
    let mut final_rows = Vec::with_capacity(rows.len());
    let mut final_cols = Vec::with_capacity(cols.len());
    let mut final_vals: Vec<f64> = Vec::with_capacity(vals.len());

    // First, add boundary identity rows
    for ni in 0..n_nodes {
        if bc_set[ni] {
            final_rows.push(ni as i64);
            final_cols.push(ni as i64);
            final_vals.push(1.0);
            load[ni] = bc_vals_map[ni];
        }
    }

    // Add interior entries, adjusting RHS for boundary neighbors
    for ((row_k, col_k), val_k) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        let ri = *row_k as usize;
        let ci = *col_k as usize;

        if bc_set[ri] {
            continue; // Skip boundary rows (already added identity)
        }

        if bc_set[ci] {
            // This column is a boundary node: move to RHS
            load[ri] -= val_k * bc_vals_map[ci];
        } else {
            final_rows.push(*row_k);
            final_cols.push(*col_k);
            final_vals.push(*val_k);
        }
    }

    // Build CSR and solve
    let nnz = final_rows.len();
    let row_t = Tensor::<R>::from_slice(&final_rows, &[nnz], device)?;
    let col_t = Tensor::<R>::from_slice(&final_cols, &[nnz], device)?;
    let val_t = Tensor::<R>::from_slice(&final_vals, &[nnz], device)?;

    // Sort and build CSR
    let csr = coo_to_csr_with_dup::<R>(&row_t, &col_t, &val_t, [n_nodes, n_nodes], device)?;

    let rhs_tensor = Tensor::<R>::from_slice(&load, &[n_nodes], device)?;

    let result = solve_sparse_system(client, &csr, &rhs_tensor, options, "FEM triangular solver")?;

    Ok(FemResult {
        solution: result.solution,
        nodes: nodes.clone(),
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

/// Convert COO with duplicate entries to CSR (sum duplicates).
fn coo_to_csr_with_dup<R: Runtime<DType = DType>>(
    row_indices: &Tensor<R>,
    col_indices: &Tensor<R>,
    values: &Tensor<R>,
    shape: [usize; 2],
    device: &R::Device,
) -> Result<CsrData<R>> {
    let [nrows, _ncols] = shape;
    let rows_vec: Vec<i64> = row_indices.to_vec();
    let cols_vec: Vec<i64> = col_indices.to_vec();
    let vals_vec: Vec<f64> = values.to_vec();
    let nnz = rows_vec.len();

    // Sort entries by (row, col) and sum duplicates
    let mut entries: Vec<(i64, i64, f64)> = (0..nnz)
        .map(|i| (rows_vec[i], cols_vec[i], vals_vec[i]))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Merge duplicates
    let mut merged_cols = Vec::with_capacity(nnz);
    let mut merged_vals: Vec<f64> = Vec::with_capacity(nnz);
    let mut row_ptrs = vec![0i64; nrows + 1];

    if !entries.is_empty() {
        let mut cur_row = entries[0].0;
        let mut cur_col = entries[0].1;
        let mut cur_val = entries[0].2;

        for &(r, c, v) in entries.iter().skip(1) {
            if r == cur_row && c == cur_col {
                cur_val += v;
            } else {
                merged_cols.push(cur_col);
                merged_vals.push(cur_val);
                row_ptrs[cur_row as usize + 1] += 1;
                cur_row = r;
                cur_col = c;
                cur_val = v;
            }
        }
        // Last entry
        merged_cols.push(cur_col);
        merged_vals.push(cur_val);
        row_ptrs[cur_row as usize + 1] += 1;
    }

    // Cumulative sum
    for i in 1..=nrows {
        row_ptrs[i] += row_ptrs[i - 1];
    }

    let final_nnz = merged_cols.len();
    let rp = Tensor::<R>::from_slice(&row_ptrs, &[nrows + 1], device)?;
    let ci = Tensor::<R>::from_slice(&merged_cols, &[final_nnz], device)?;
    let vv = Tensor::<R>::from_slice(&merged_vals, &[final_nnz], device)?;

    CsrData::new(rp, ci, vv, shape)
}
