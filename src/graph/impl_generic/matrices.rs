//! Graph matrix construction (Laplacian, Adjacency, Incidence).
//!
//! GPU-parallel via sparse operations.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::{SparseOps, SparseTensor};
use numr::tensor::Tensor;

use crate::graph::traits::types::GraphData;

use super::helpers::extract_csr_arrays;

/// Compute the graph Laplacian matrix L = D - A.
///
/// If normalized: L_norm = I - D^{-1/2} A D^{-1/2}.
pub fn laplacian_matrix_impl<R, C>(
    _client: &C,
    graph: &GraphData<R>,
    normalized: bool,
) -> Result<SparseTensor<R>>
where
    R: Runtime,
    C: SparseOps<R>,
{
    // Extract CSR
    let (row_ptrs, col_indices, _values, n) = extract_csr_arrays(graph)?;

    // Get device
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Compute degree for each node (edge count for standard Laplacian)
    let degrees: Vec<f64> = (0..n)
        .map(|u| {
            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            (end - start) as f64
        })
        .collect();

    if !normalized {
        // L = D - A
        // Build L as CSR format directly

        let mut lap_row_ptrs = Vec::with_capacity(n + 1);
        let mut lap_col_indices = Vec::new();
        let mut lap_values = Vec::new();

        lap_row_ptrs.push(0);

        for u in 0..n {
            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;

            // Add diagonal entry: degrees[u]
            lap_col_indices.push(u as i64);
            lap_values.push(degrees[u]);

            // Add off-diagonal entries: -1 for each edge (negated adjacency)
            for &v_idx in col_indices.iter().take(end).skip(start) {
                let v = v_idx as usize;
                if v != u {
                    lap_col_indices.push(v as i64);
                    lap_values.push(-1.0);
                }
            }

            lap_row_ptrs.push(lap_col_indices.len() as i64);
        }

        // Convert to CSR
        let row_ptrs_tensor = Tensor::<R>::from_slice(&lap_row_ptrs, &[n + 1], &device);
        let col_indices_tensor =
            Tensor::<R>::from_slice(&lap_col_indices, &[lap_col_indices.len()], &device);
        let values_tensor = Tensor::<R>::from_slice(&lap_values, &[lap_values.len()], &device);

        let csr =
            numr::sparse::CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])?;
        Ok(SparseTensor::Csr(csr))
    } else {
        // Normalized: L_norm = I - D^{-1/2} A D^{-1/2}
        // Build as dense then convert to sparse (simple approach)

        let mut lap = vec![vec![0.0; n]; n];

        // Compute D^{-1/2}
        let deg_inv_sqrt: Vec<f64> = degrees
            .iter()
            .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 0.0 })
            .collect();

        // L_norm[i][j] = delta_ij - d_sqrt[i] * A[i][j] * d_sqrt[j]
        for u in 0..n {
            lap[u][u] = 1.0; // Diagonal

            let start = row_ptrs[u] as usize;
            let end = row_ptrs[u + 1] as usize;
            for &v_idx in col_indices.iter().take(end).skip(start) {
                let v = v_idx as usize;
                if u != v {
                    lap[u][v] -= deg_inv_sqrt[u] * deg_inv_sqrt[v];
                }
            }
        }

        // Convert to sparse (COO then CSR)
        let mut coo_rows = Vec::new();
        let mut coo_cols = Vec::new();
        let mut coo_vals = Vec::new();

        for (i, row) in lap.iter().enumerate().take(n) {
            for (j, &val) in row.iter().enumerate().take(n) {
                if val.abs() > 1e-15 {
                    coo_rows.push(i as i64);
                    coo_cols.push(j as i64);
                    coo_vals.push(val);
                }
            }
        }

        let sparse =
            SparseTensor::<R>::from_coo_slices(&coo_rows, &coo_cols, &coo_vals, [n, n], &device)?;

        sparse.to_csr()
    }
}

/// Return the adjacency matrix (possibly symmetrized for undirected graphs).
pub fn adjacency_matrix_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<SparseTensor<R>>
where
    R: Runtime,
    C: SparseOps<R>,
{
    // For directed graphs, return as-is.
    // For undirected graphs, the adjacency should already be symmetric in CSR.
    Ok(graph.adjacency.clone())
}

/// Compute the incidence matrix B.
///
/// B[i, e] = -1 if edge e leaves node i, +1 if edge e enters node i.
/// For undirected graphs, arbitrary orientation is assigned.
pub fn incidence_matrix_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<SparseTensor<R>>
where
    R: Runtime,
    C: SparseOps<R>,
{
    // Extract CSR
    let (row_ptrs, col_indices, _values, n) = extract_csr_arrays(graph)?;

    // Get device
    let device = match &graph.adjacency {
        SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Extract edge list from CSR (one pass)
    let mut edge_list = Vec::new();
    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for &v_idx in col_indices.iter().take(end).skip(start) {
            let v = v_idx as usize;
            // For undirected graphs, only take u < v to avoid duplicate edges
            if graph.directed || u < v {
                edge_list.push((u, v));
            }
        }
    }

    let m = edge_list.len(); // Number of edges
    if m == 0 {
        // Empty incidence matrix
        return SparseTensor::<R>::from_coo_slices::<f64>(&[], &[], &[], [n, 1], &device);
    }

    // Build incidence matrix: B[i, e]
    // For each edge e = (u, v):
    //   B[u, e] = -1 (edge leaves u)
    //   B[v, e] = +1 (edge enters v)
    let mut inc_rows = Vec::new();
    let mut inc_cols = Vec::new();
    let mut inc_vals = Vec::new();

    for (e, (u, v)) in edge_list.iter().enumerate() {
        let e = e as i64;
        // Edge leaves u
        inc_rows.push(*u as i64);
        inc_cols.push(e);
        inc_vals.push(-1.0);

        // Edge enters v
        inc_rows.push(*v as i64);
        inc_cols.push(e);
        inc_vals.push(1.0);
    }

    SparseTensor::<R>::from_coo_slices(&inc_rows, &inc_cols, &inc_vals, [n, m], &device)?.to_csr()
}
