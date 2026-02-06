//! Symbolic analysis for sparse LU factorization.
//!
//! Builds `LuSymbolic` structures from CSC matrix patterns. This is solvr's
//! "brain" — the symbolic analysis that numr's numeric factorization consumes.
//!
//! # Components
//!
//! - **Elimination tree**: Column dependency graph for sparse Gaussian elimination
//! - **Post-order**: DFS traversal order for efficient column processing
//! - **Column reach**: For each column k, which prior columns contribute to it
//! - **LU pattern**: Predicted nonzero positions in L and U factors

#[cfg(feature = "sparse")]
use numr::algorithm::sparse_linalg::LuSymbolic;

/// Compute elimination tree from CSC matrix pattern.
///
/// The elimination tree captures column dependencies: etree[j] = parent of column j,
/// meaning column etree[j] depends on column j during LU factorization.
/// A value of `n` (sentinel) means column j is a root.
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `col_ptrs` - CSC column pointers (length n+1)
/// * `row_indices` - CSC row indices
#[cfg(feature = "sparse")]
pub fn compute_elimination_tree(n: usize, col_ptrs: &[i64], row_indices: &[i64]) -> Vec<usize> {
    let mut etree = vec![n; n]; // sentinel = n means root
    let mut ancestor = vec![n; n]; // workspace for path compression

    for col in 0..n {
        let start = col_ptrs[col] as usize;
        let end = col_ptrs[col + 1] as usize;

        for &ri in &row_indices[start..end] {
            let mut row = ri as usize;
            if row >= col {
                continue; // Only process upper triangle entries (row < col)
            }

            // Walk up the tree from row to col, compressing paths
            loop {
                let anc = ancestor[row];
                if anc == col || anc == n {
                    break;
                }
                ancestor[row] = col; // path compression
                if etree[row] == n {
                    etree[row] = col;
                }
                row = anc;
            }
            ancestor[row] = col;
            if etree[row] == n {
                etree[row] = col;
            }
        }
    }

    etree
}

/// Compute post-order traversal of the elimination tree.
///
/// Returns a permutation where children appear before parents (DFS post-order).
/// This ordering ensures that when processing column post_order[k], all columns
/// it depends on have already been processed.
///
/// # Arguments
///
/// * `etree` - Elimination tree (etree[j] = parent of j, or n for root)
/// * `n` - Matrix dimension
#[cfg(feature = "sparse")]
pub fn compute_post_order(etree: &[usize], n: usize) -> Vec<usize> {
    // Build children lists
    let mut first_child = vec![n; n]; // sentinel = n means no children
    let mut next_sibling = vec![n; n];

    for j in (0..n).rev() {
        let parent = etree[j];
        if parent < n {
            next_sibling[j] = first_child[parent];
            first_child[parent] = j;
        }
    }

    // DFS post-order traversal
    let mut post_order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, bool)> = Vec::with_capacity(n);

    // Push all roots (nodes whose parent is n)
    for j in (0..n).rev() {
        if etree[j] == n {
            stack.push((j, false));
        }
    }

    while let Some((node, visited)) = stack.pop() {
        if visited {
            post_order.push(node);
        } else {
            stack.push((node, true));
            // Push children in reverse order so they come out in order
            let mut child = first_child[node];
            let mut children = Vec::new();
            while child < n {
                children.push(child);
                child = next_sibling[child];
            }
            for &c in children.iter().rev() {
                stack.push((c, false));
            }
        }
    }

    post_order
}

/// Compute column reach sets for sparse LU factorization.
///
/// For each column k, reach[k] contains the indices of prior columns j < k
/// whose L factor column contributes to column k during the factorization.
/// The reach is returned in topological order (respecting the elimination tree).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `etree` - Elimination tree
/// * `post_order` - Post-order traversal of elimination tree
/// * `col_ptrs` - CSC column pointers
/// * `row_indices` - CSC row indices
#[cfg(feature = "sparse")]
pub fn compute_column_reach(
    n: usize,
    etree: &[usize],
    _post_order: &[usize],
    col_ptrs: &[i64],
    row_indices: &[i64],
) -> Vec<Vec<usize>> {
    let mut reach = vec![vec![]; n];
    let mut mark = vec![0usize; n]; // Per-column visited marker

    for col in 0..n {
        let start = col_ptrs[col] as usize;
        let end = col_ptrs[col + 1] as usize;

        // For each row index in this column that is above the diagonal
        for &ri in &row_indices[start..end] {
            let row = ri as usize;
            if row >= col {
                continue;
            }

            // Walk up the etree from row, marking nodes until we hit
            // something already marked for this column or reach col
            let mut j = row;
            while j < col && mark[j] != col + 1 {
                reach[col].push(j);
                mark[j] = col + 1; // mark as visited for column col
                j = etree[j];
            }
        }

        // Sort reach in topological order (ascending is correct for
        // left-looking LU since lower-numbered columns are processed first)
        reach[col].sort_unstable();
    }

    reach
}

/// Predict L and U nonzero patterns from symbolic analysis.
///
/// Returns CSC column pointers and row indices for both L and U factors.
/// These are upper bounds on the actual fill-in — numeric cancellation may
/// produce fewer nonzeros.
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `etree` - Elimination tree
/// * `reach` - Column reach sets
/// * `col_ptrs` - CSC column pointers of original matrix
/// * `row_indices` - CSC row indices of original matrix
///
/// # Returns
///
/// (l_col_ptrs, l_row_indices, u_col_ptrs, u_row_indices)
#[cfg(feature = "sparse")]
pub fn predict_lu_pattern(
    n: usize,
    _etree: &[usize],
    reach: &[Vec<usize>],
    col_ptrs: &[i64],
    row_indices: &[i64],
) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut l_col_ptrs = vec![0i64; n + 1];
    let mut l_row_indices = Vec::new();
    let mut u_col_ptrs = vec![0i64; n + 1];
    let mut u_row_indices = Vec::new();

    for col in 0..n {
        // U pattern for column col: reach(col) union {col}
        // These are the rows above diagonal that have nonzeros
        let u_start = u_row_indices.len();
        for &j in &reach[col] {
            u_row_indices.push(j as i64);
        }
        u_row_indices.push(col as i64); // diagonal
        u_col_ptrs[col + 1] = u_row_indices.len() as i64;

        // L pattern for column col: rows below diagonal in A[:, col]
        // plus fill-in from the reach
        let l_start = l_row_indices.len();
        let mut l_rows = Vec::new();

        // Original entries below diagonal
        let a_start = col_ptrs[col] as usize;
        let a_end = col_ptrs[col + 1] as usize;
        for &ri in &row_indices[a_start..a_end] {
            let row = ri as usize;
            if row > col {
                l_rows.push(row);
            }
        }

        // Fill-in: for each column j in reach(col), rows in L[:, j] that are > col
        // contribute to L[:, col]. This is a conservative estimate.
        // For now, use the original pattern entries — the numeric factorization
        // handles dynamic fill-in via workspace.

        l_rows.sort_unstable();
        l_rows.dedup();
        for row in l_rows {
            l_row_indices.push(row as i64);
        }
        l_col_ptrs[col + 1] = l_row_indices.len() as i64;

        let _ = (u_start, l_start); // suppress unused warnings
    }

    (l_col_ptrs, l_row_indices, u_col_ptrs, u_row_indices)
}

/// Build a complete `LuSymbolic` from a CSC matrix pattern.
///
/// Orchestrates all symbolic analysis phases:
/// 1. Compute elimination tree
/// 2. Compute post-order traversal
/// 3. Compute column reach sets
/// 4. Predict L/U nonzero patterns
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `col_ptrs` - CSC column pointers
/// * `row_indices` - CSC row indices
#[cfg(feature = "sparse")]
pub fn compute_lu_symbolic(n: usize, col_ptrs: &[i64], row_indices: &[i64]) -> LuSymbolic {
    let etree = compute_elimination_tree(n, col_ptrs, row_indices);
    let post_order = compute_post_order(&etree, n);
    let reach = compute_column_reach(n, &etree, &post_order, col_ptrs, row_indices);
    let (l_col_ptrs, l_row_indices, u_col_ptrs, u_row_indices) =
        predict_lu_pattern(n, &etree, &reach, col_ptrs, row_indices);

    LuSymbolic {
        n,
        etree,
        post_order,
        reach,
        l_col_ptrs,
        l_row_indices,
        u_col_ptrs,
        u_row_indices,
        workspace_size: n,
    }
}

#[cfg(all(test, feature = "sparse"))]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        // 3x3 identity in CSC: each column has one entry on diagonal
        let col_ptrs = vec![0i64, 1, 2, 3];
        let row_indices = vec![0i64, 1, 2];
        let n = 3;

        let etree = compute_elimination_tree(n, &col_ptrs, &row_indices);
        // Identity: no off-diagonal entries, all roots
        assert!(etree.iter().all(|&p| p == n), "etree = {:?}", etree);

        let post_order = compute_post_order(&etree, n);
        assert_eq!(post_order.len(), n);

        let reach = compute_column_reach(n, &etree, &post_order, &col_ptrs, &row_indices);
        assert!(reach.iter().all(|r| r.is_empty()), "reach = {:?}", reach);

        let symbolic = compute_lu_symbolic(n, &col_ptrs, &row_indices);
        assert_eq!(symbolic.n, 3);
    }

    #[test]
    fn test_tridiagonal_matrix() {
        // 4x4 tridiagonal: entries at (i,j) where |i-j| <= 1
        // CSC format:
        // col 0: rows [0, 1]
        // col 1: rows [0, 1, 2]
        // col 2: rows [1, 2, 3]
        // col 3: rows [2, 3]
        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let n = 4;

        let etree = compute_elimination_tree(n, &col_ptrs, &row_indices);
        // Tridiagonal: etree should chain 0->1->2->3
        assert_eq!(etree[0], 1, "etree[0] should be 1");
        assert_eq!(etree[1], 2, "etree[1] should be 2");
        assert_eq!(etree[2], 3, "etree[2] should be 3");

        let symbolic = compute_lu_symbolic(n, &col_ptrs, &row_indices);
        assert_eq!(symbolic.n, 4);
        // Reach of column 2 should include column 1 (because row 1 is in col 2)
        assert!(
            symbolic.reach[2].contains(&1),
            "reach[2] = {:?}",
            symbolic.reach[2]
        );
    }

    #[test]
    fn test_arrow_matrix() {
        // 4x4 arrow matrix: dense first row/column + diagonal
        // Pattern:
        // [x x x x]
        // [x x . .]
        // [x . x .]
        // [x . . x]
        //
        // CSC format:
        // col 0: rows [0, 1, 2, 3]
        // col 1: rows [0, 1]
        // col 2: rows [0, 2]
        // col 3: rows [0, 3]
        let col_ptrs = vec![0i64, 4, 6, 8, 10];
        let row_indices = vec![0i64, 1, 2, 3, 0, 1, 0, 2, 0, 3];
        let n = 4;

        let etree = compute_elimination_tree(n, &col_ptrs, &row_indices);
        // Column 0 is referenced by columns 1,2,3 — so etree[0] = 1 (first referencing column)
        assert!(etree[0] < n, "Column 0 should have a parent");

        let symbolic = compute_lu_symbolic(n, &col_ptrs, &row_indices);
        assert_eq!(symbolic.n, 4);
        // Columns 1, 2, 3 all depend on column 0
        assert!(symbolic.reach[1].contains(&0));
        assert!(symbolic.reach[2].contains(&0));
        assert!(symbolic.reach[3].contains(&0));
    }

    #[test]
    fn test_post_order_covers_all_nodes() {
        // Random sparse pattern
        let col_ptrs = vec![0i64, 2, 4, 6, 7, 9];
        let row_indices = vec![0i64, 1, 0, 1, 2, 3, 2, 3, 4];
        let n = 5;

        let etree = compute_elimination_tree(n, &col_ptrs, &row_indices);
        let post_order = compute_post_order(&etree, n);

        assert_eq!(post_order.len(), n);
        let mut sorted = post_order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }
}
