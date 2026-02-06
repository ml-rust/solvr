//! Direct sparse LU solver for implicit ODE solvers.
//!
//! `DirectSparseSolver<R>` manages the full lifecycle of sparse LU factorization
//! with symbolic caching. It caches the column ordering (COLAMD), symbolic
//! analysis, CSC pattern, gather indices, and LU workspace across Newton
//! iterations and time steps.
//!
//! # Lifecycle
//!
//! ```text
//! First call:    dense→CSR→CSC → COLAMD → Hopcroft-Karp (if needed)
//!                → LuSymbolic → cache CSC pattern + gather indices
//!                → LuWorkspace → numeric LU → solve
//! Subsequent:    gather_2d(dense, cached_indices) → update_values(CSC)
//!                → numeric LU with workspace reuse → solve
//! Invalidate:    force full re-analysis on next call
//! ```

#[cfg(feature = "sparse")]
use numr::algorithm::sparse_linalg::{
    ColamdOptions, LuFactors, LuOptions, LuSymbolic, LuWorkspace, colamd, hopcroft_karp,
    sparse_lu_cpu_with_workspace_and_metrics, sparse_lu_solve_cpu,
};
#[cfg(feature = "sparse")]
use numr::error::Result;
#[cfg(feature = "sparse")]
use numr::ops::IndexingOps;
#[cfg(feature = "sparse")]
use numr::runtime::Runtime;
#[cfg(feature = "sparse")]
use numr::sparse::{CscData, SparseOps, SparseScaling, SparseStorage};
#[cfg(feature = "sparse")]
use numr::tensor::Tensor;

#[cfg(feature = "sparse")]
use super::direct_solver_config::DirectSolverConfig;
#[cfg(feature = "sparse")]
use super::sparse_utils::dense_to_csr_full;
#[cfg(feature = "sparse")]
use super::symbolic_analysis::compute_lu_symbolic;

/// Direct sparse LU solver with symbolic caching and optimized value extraction.
///
/// On the first solve, performs full analysis (COLAMD ordering, symbolic
/// factorization, workspace allocation) and caches the CSC pattern plus
/// gather indices. On subsequent solves, uses `gather_2d` to extract values
/// at cached nonzero positions in O(nnz) time, avoiding the O(n²) dense→CSR→CSC
/// conversion.
///
/// # Performance
///
/// - **First solve**: O(n²) dense→CSR→CSC + O(nnz) ordering + O(n) symbolic + O(nnz) numeric
/// - **Subsequent solves**: O(nnz) gather + O(nnz) numeric (no allocation)
/// - **Typical speedup**: 5-20x for subsequent solves vs full conversion each time
///
/// # CSC CPU Residency
///
/// Sparse LU operations (COLAMD, symbolic analysis, factorization) are CPU-based
/// in numr. CSC metadata (col_ptrs, row_indices) is therefore pulled to CPU during
/// `full_analysis()`, `build_gather_indices()`, and `permute_csc_columns()`. This
/// is architecturally justified—no GPU transfer overhead is incurred.
#[cfg(feature = "sparse")]
pub struct DirectSparseSolver<R: Runtime> {
    /// Cached column permutation from COLAMD
    col_perm: Option<Vec<usize>>,
    /// Cached row permutation from Hopcroft-Karp (only if diagonal has zeros) - CPU version
    row_perm: Option<Vec<usize>>,
    /// Row permutation as I64 tensor for on-device index_select
    row_perm_tensor: Option<Tensor<R>>,
    /// Inverse column permutation as I64 tensor for on-device index_select
    inv_col_perm_tensor: Option<Tensor<R>>,

    /// Cached symbolic analysis
    symbolic: Option<LuSymbolic>,
    /// Cached numeric factors (stored for potential reuse)
    factors: Option<LuFactors<R>>,
    /// LU factorization options (derived from DirectSolverConfig)
    lu_options: LuOptions,

    /// Cached permuted CSC structure for fast value updates.
    /// On subsequent solves, only values are replaced via `update_values`.
    cached_permuted_csc: Option<CscData<R>>,

    /// Gather row indices (I64 tensor [nnz]) mapping CSC nonzeros to dense matrix rows.
    gather_row_indices: Option<Tensor<R>>,
    /// Gather column indices (I64 tensor [nnz]) mapping CSC nonzeros to dense matrix columns.
    gather_col_indices: Option<Tensor<R>>,

    /// Pre-allocated workspace for numeric LU refactorization (no allocation after first solve).
    workspace: Option<LuWorkspace>,

    /// Cached row scales from equilibration (1/row_norms) - CPU version for CSC scale_rows
    row_scales: Option<Vec<f64>>,
    /// Cached column scales from equilibration (1/col_norms) - CPU version for CSC scale_cols
    col_scales: Option<Vec<f64>>,
    /// Row scales as tensor for on-device b scaling
    row_scales_tensor: Option<Tensor<R>>,
    /// Column scales as tensor for on-device solution scaling
    col_scales_tensor: Option<Tensor<R>>,

    /// Whether equilibration is enabled.
    equilibrate: bool,
    /// Pivot growth threshold for diagnostics.
    pivot_growth_threshold: f64,

    /// Number of times numeric refactorization was performed.
    pub refactor_count: usize,
    /// Number of times full re-ordering was performed.
    pub reorder_count: usize,
    /// Pivot growth factor from the last factorization.
    pub last_pivot_growth: f64,
    /// Number of small pivots from the last factorization.
    pub last_small_pivots: usize,
}

#[cfg(feature = "sparse")]
impl<R: Runtime> DirectSparseSolver<R> {
    /// Create a new direct sparse solver with the given configuration.
    pub fn new(config: &DirectSolverConfig) -> Self {
        Self {
            col_perm: None,
            row_perm: None,
            row_perm_tensor: None,
            inv_col_perm_tensor: None,
            symbolic: None,
            factors: None,
            lu_options: LuOptions {
                pivot_tolerance: config.pivot_tolerance,
                pivot_threshold: config.pivot_threshold,
                diagonal_shift: config.diagonal_shift,
                check_zeros: true,
            },
            cached_permuted_csc: None,
            gather_row_indices: None,
            gather_col_indices: None,
            workspace: None,
            row_scales: None,
            col_scales: None,
            row_scales_tensor: None,
            col_scales_tensor: None,
            equilibrate: config.equilibrate,
            pivot_growth_threshold: config.pivot_growth_threshold,
            refactor_count: 0,
            reorder_count: 0,
            last_pivot_growth: 0.0,
            last_small_pivots: 0,
        }
    }

    /// Solve M*x = b using direct sparse LU factorization.
    ///
    /// On the first call, performs full analysis: dense→CSR→CSC, COLAMD ordering,
    /// symbolic analysis, gather index construction, and workspace allocation.
    /// On subsequent calls, uses `gather_2d` to extract values at cached nonzero
    /// positions and `update_values` to update the CSC in-place.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client with sparse and indexing operations
    /// * `m_dense` - Dense iteration matrix (I - hγJ)
    /// * `b` - Right-hand side vector [n] or [n, 1]
    pub fn solve<C>(&mut self, client: &C, m_dense: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: SparseOps<R> + IndexingOps<R> + numr::ops::TensorOps<R> + numr::ops::ScalarOps<R>,
    {
        if self.symbolic.is_none() {
            // First call: full analysis + cache pattern + build gather indices + workspace
            let csr = dense_to_csr_full(client, m_dense)?;
            let csc = csr.to_csc()?;
            let n = csc.shape()[0];
            self.full_analysis(&csc, n)?;
        } else {
            // Subsequent calls: gather values at cached positions, update CSC in-place
            let values = client.gather_2d(
                m_dense,
                self.gather_row_indices
                    .as_ref()
                    .expect("gather indices set after full_analysis"),
                self.gather_col_indices
                    .as_ref()
                    .expect("gather indices set after full_analysis"),
            )?;
            self.cached_permuted_csc
                .as_mut()
                .expect("cached CSC set after full_analysis")
                .update_values(values)?;
        }

        let n = self.cached_permuted_csc.as_ref().unwrap().shape()[0];
        let symbolic = self.symbolic.as_ref().unwrap();
        let workspace = self.workspace.as_mut().unwrap();

        // Apply equilibration to cached CSC if enabled
        let factored_csc = if self.equilibrate {
            let csc_ref = self.cached_permuted_csc.as_ref().unwrap();
            if self.row_scales.is_some() {
                // Reuse cached scales
                let row_scales = self.row_scales.as_ref().unwrap();
                let col_scales = self.col_scales.as_ref().unwrap();
                let scaled = csc_ref.scale_rows(row_scales)?;
                scaled.scale_cols(col_scales)?
            } else {
                // Should not happen — scales are computed during full_analysis
                csc_ref.clone()
            }
        } else {
            self.cached_permuted_csc.as_ref().unwrap().clone()
        };

        // Numeric factorization with workspace reuse
        let (factors, metrics) = sparse_lu_cpu_with_workspace_and_metrics(
            &factored_csc,
            symbolic,
            &self.lu_options,
            workspace,
        )?;

        // Track diagnostics
        self.last_pivot_growth = metrics.pivot_growth;
        self.last_small_pivots = metrics.small_pivots;
        self.refactor_count += 1;

        // Solve + permute back (using on-device tensor ops)
        let solution = self.solve_with_factors(client, &factors, b, n)?;
        self.factors = Some(factors);

        Ok(solution)
    }

    /// Perform full symbolic analysis: COLAMD ordering + optional Hopcroft-Karp + etree.
    ///
    /// Also caches the permuted CSC pattern, builds gather indices for subsequent
    /// `gather_2d` calls, allocates the LU workspace, and optionally computes
    /// equilibration scales.
    fn full_analysis(&mut self, csc: &CscData<R>, n: usize) -> Result<()> {
        let col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let row_indices: Vec<i64> = csc.row_indices().to_vec();

        // Step 1: COLAMD column ordering to reduce fill-in
        let colamd_opts = ColamdOptions::default();
        let (col_perm, _stats) = colamd(n, n, &col_ptrs, &row_indices, &colamd_opts)?;

        // Step 2: Permute columns
        let perm_csc = Self::permute_csc_columns(csc, &col_perm, n)?;

        // Step 3: Check diagonal and optionally compute row permutation
        let needs_row_perm = !perm_csc.has_full_diagonal();

        if needs_row_perm {
            let perm_col_ptrs: Vec<i64> = perm_csc.col_ptrs().to_vec();
            let perm_row_indices: Vec<i64> = perm_csc.row_indices().to_vec();

            // Hopcroft-Karp for row permutation to maximize diagonal nonzeros
            let matching = hopcroft_karp(n, n, &perm_col_ptrs, &perm_row_indices)?;

            // Convert matching to row permutation
            let mut row_perm = vec![0usize; n];
            for (col, &row) in matching.col_to_row.iter().enumerate() {
                if row >= 0 && (row as usize) < n {
                    row_perm[col] = row as usize;
                } else {
                    row_perm[col] = col; // unmatched, keep in place
                }
            }
            self.row_perm = Some(row_perm);

            // Compute symbolic on the column-permuted pattern
            let symbolic = compute_lu_symbolic(n, &perm_col_ptrs, &perm_row_indices);
            self.workspace = Some(LuWorkspace::new(n, &symbolic));
            self.symbolic = Some(symbolic);
        } else {
            let perm_col_ptrs: Vec<i64> = perm_csc.col_ptrs().to_vec();
            let perm_row_indices: Vec<i64> = perm_csc.row_indices().to_vec();

            let symbolic = compute_lu_symbolic(n, &perm_col_ptrs, &perm_row_indices);
            self.workspace = Some(LuWorkspace::new(n, &symbolic));
            self.symbolic = Some(symbolic);
        }

        // Step 4: Optional equilibration — compute and cache scales (both CPU and tensor versions)
        let device = perm_csc.col_ptrs().device();
        if self.equilibrate {
            let (_scaled, row_scales, col_scales) = perm_csc.equilibrate::<f64>()?;
            // CPU versions for CSC scale_rows/scale_cols
            self.row_scales = Some(row_scales.clone());
            self.col_scales = Some(col_scales.clone());
            // Tensor versions for on-device b/solution scaling
            self.row_scales_tensor = Some(Tensor::<R>::from_slice(&row_scales, &[n], device));
            self.col_scales_tensor = Some(Tensor::<R>::from_slice(&col_scales, &[n], device));
        }

        // Step 5: Build gather indices mapping permuted CSC nonzeros → dense matrix positions
        let (gather_rows, gather_cols) = Self::build_gather_indices(&perm_csc, n, &col_perm)?;
        self.gather_row_indices = Some(gather_rows);
        self.gather_col_indices = Some(gather_cols);

        // Step 6: Build tensor permutations for on-device solve_with_factors
        if let Some(row_perm) = &self.row_perm {
            let row_perm_i64: Vec<i64> = row_perm.iter().map(|&i| i as i64).collect();
            self.row_perm_tensor = Some(Tensor::<R>::from_slice(&row_perm_i64, &[n], device));
        }
        // Compute inverse column permutation: inv[col_perm[i]] = i
        let mut inv_col_perm = vec![0usize; n];
        for i in 0..n {
            inv_col_perm[col_perm[i]] = i;
        }
        let inv_col_perm_i64: Vec<i64> = inv_col_perm.iter().map(|&i| i as i64).collect();
        self.inv_col_perm_tensor = Some(Tensor::<R>::from_slice(&inv_col_perm_i64, &[n], device));

        // Step 7: Cache the permuted CSC (subsequent solves update values in-place)
        self.cached_permuted_csc = Some(perm_csc);

        self.col_perm = Some(col_perm);
        self.reorder_count += 1;

        Ok(())
    }

    /// Build I64 gather index tensors mapping each permuted CSC nonzero to its
    /// (row, col) position in the original dense matrix.
    ///
    /// The permuted CSC has columns reordered by `col_perm`. For `gather_2d`,
    /// we need to know where each nonzero came from in the original dense matrix:
    /// - Row index is unchanged (CSC column permutation doesn't change row indices)
    /// - Column index maps back through `col_perm`: permuted col j → original col `col_perm[j]`
    fn build_gather_indices(
        perm_csc: &CscData<R>,
        n: usize,
        col_perm: &[usize],
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let col_ptrs: Vec<i64> = perm_csc.col_ptrs().to_vec();
        let row_indices: Vec<i64> = perm_csc.row_indices().to_vec();
        let nnz = row_indices.len();

        let mut dense_rows = Vec::with_capacity(nnz);
        let mut dense_cols = Vec::with_capacity(nnz);

        for perm_col in 0..n {
            let orig_col = col_perm[perm_col];
            let start = col_ptrs[perm_col] as usize;
            let end = col_ptrs[perm_col + 1] as usize;
            for &ri in &row_indices[start..end] {
                dense_rows.push(ri); // row unchanged
                dense_cols.push(orig_col as i64); // map back to original column
            }
        }

        let device = perm_csc.col_ptrs().device();
        let row_tensor = Tensor::<R>::from_slice(&dense_rows, &[nnz], device);
        let col_tensor = Tensor::<R>::from_slice(&dense_cols, &[nnz], device);
        Ok((row_tensor, col_tensor))
    }

    /// Permute CSC columns according to a column permutation.
    ///
    /// Given permutation perm, the new column j gets the old column perm[j].
    fn permute_csc_columns(csc: &CscData<R>, col_perm: &[usize], n: usize) -> Result<CscData<R>> {
        let old_col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let old_row_indices: Vec<i64> = csc.row_indices().to_vec();
        let old_values: Vec<f64> = csc.values().to_vec();

        let mut new_col_ptrs = vec![0i64; n + 1];
        let mut new_row_indices = Vec::new();
        let mut new_values = Vec::new();

        for new_col in 0..n {
            let old_col = col_perm[new_col];
            let start = old_col_ptrs[old_col] as usize;
            let end = old_col_ptrs[old_col + 1] as usize;

            for (&ri, &val) in old_row_indices[start..end]
                .iter()
                .zip(&old_values[start..end])
            {
                new_row_indices.push(ri);
                new_values.push(val);
            }
            new_col_ptrs[new_col + 1] = new_row_indices.len() as i64;
        }

        let device = csc.col_ptrs().device();
        CscData::from_slices(&new_col_ptrs, &new_row_indices, &new_values, [n, n], device)
    }

    /// Solve using LU factors, handling column/row permutations and equilibration scaling.
    ///
    /// All permutations and scaling operations are performed on-device using tensor ops.
    ///
    /// For PA[:,perm] = LU, we solve A*x = b as:
    /// 1. Permute b by row_perm (if any): b_perm = index_select(b, row_perm_tensor)
    /// 2. If equilibrated: scale b_perm by row_scales: b_scaled = b_perm * row_scales_tensor
    /// 3. Solve LU * z = b_perm (CPU operation via sparse_lu_solve_cpu)
    /// 4. If equilibrated: scale z by col_scales: z_scaled = z * col_scales_tensor
    /// 5. Inverse-permute z by col_perm: x = index_select(z, inv_col_perm_tensor)
    fn solve_with_factors<C>(
        &self,
        client: &C,
        factors: &LuFactors<R>,
        b: &Tensor<R>,
        n: usize,
    ) -> Result<Tensor<R>>
    where
        C: IndexingOps<R> + numr::ops::TensorOps<R> + numr::ops::ScalarOps<R>,
    {
        // Handle b shape: might be [n] or [n, 1]
        let b_shape = b.shape().to_vec();
        let b_flat = if b_shape.len() == 2 && b_shape[1] == 1 {
            b.reshape(&[n])?
        } else {
            b.clone()
        };

        // Apply row permutation to b if we have one (using on-device index_select)
        let b_perm = if let Some(row_perm_tensor) = &self.row_perm_tensor {
            client.index_select(&b_flat, 0, row_perm_tensor)?
        } else {
            b_flat
        };

        // Apply equilibration row scaling to b (using on-device element-wise multiply)
        let b_scaled = if self.equilibrate {
            if let Some(row_scales_tensor) = &self.row_scales_tensor {
                client.mul(&b_perm, row_scales_tensor)?
            } else {
                b_perm
            }
        } else {
            b_perm
        };

        // Forward/backward substitution with LU factors (CPU operation)
        let z = sparse_lu_solve_cpu(factors, &b_scaled)?;

        // Apply equilibration column scaling to solution (using on-device element-wise multiply)
        let z_scaled = if self.equilibrate {
            if let Some(col_scales_tensor) = &self.col_scales_tensor {
                client.mul(&z, col_scales_tensor)?
            } else {
                z
            }
        } else {
            z
        };

        // Inverse column permutation (using on-device index_select)
        let result =
            client.index_select(&z_scaled, 0, self.inv_col_perm_tensor.as_ref().unwrap())?;

        // Restore original shape if needed
        if b_shape.len() == 2 && b_shape[1] == 1 {
            result.reshape(&[n, 1])
        } else {
            Ok(result)
        }
    }

    /// Invalidate all cached analysis, forcing full re-analysis on next solve.
    ///
    /// Call this when the sparsity pattern changes (e.g., after Jacobian recomputation
    /// that might change structure).
    pub fn invalidate(&mut self) {
        self.col_perm = None;
        self.row_perm = None;
        self.row_perm_tensor = None;
        self.inv_col_perm_tensor = None;
        self.symbolic = None;
        self.factors = None;
        self.cached_permuted_csc = None;
        self.gather_row_indices = None;
        self.gather_col_indices = None;
        self.workspace = None;
        self.row_scales = None;
        self.col_scales = None;
        self.row_scales_tensor = None;
        self.col_scales_tensor = None;
    }

    /// Returns true if symbolic analysis has been computed and cached.
    pub fn has_symbolic(&self) -> bool {
        self.symbolic.is_some()
    }

    /// Returns the pivot growth factor from the last factorization.
    ///
    /// A large pivot growth (> `pivot_growth_threshold`) indicates the
    /// factorization may be numerically unreliable.
    pub fn pivot_growth_unreliable(&self) -> bool {
        self.last_pivot_growth > self.pivot_growth_threshold
    }

    /// Returns the last LU factorization metrics, if available.
    pub fn last_metrics(&self) -> Option<(f64, usize)> {
        if self.refactor_count > 0 {
            Some((self.last_pivot_growth, self.last_small_pivots))
        } else {
            None
        }
    }
}
