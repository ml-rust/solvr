//! Tensor-based simplex method for linear programming.
//!
//! # Architecture
//!
//! The tableau is stored as a 2D `Tensor<R>` and the critical **pivot operations**
//! use tensor broadcasting for efficient row elimination. This is where the main
//! computational work happens (O(n_iterations * tableau_size)).
//!
//! ## Tensor Operations (GPU-accelerated):
//! - `pivot_tensor`: Uses matmul for outer product, tensor subtraction for elimination
//! - Tableau storage: Kept on device throughout iterations
//!
//! ## Acceptable CPU Operations:
//! - **Initial setup** (one-time): Tableau construction from constraint data
//! - **Control flow** (per iteration): Extracting single row/column for pivot selection
//!   - argmin over objective row → O(n_vars) extract
//!   - ratio test over pivot column → O(n_constraints) extract
//!   These are small extracts for branching decisions, not bulk computation.
//!
//! ## Completed Optimizations:
//! - Pivot column selection uses `argmin` over objective row
//! - Pivot row selection uses `where_cond` + `argmin` for ratio test
//!
//! ## Future Improvements:
//! - Build initial tableau using tensor concatenation instead of scalar loops

use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::linprog::LinProgOptions;

use super::{TensorLinProgResult, TensorLinearConstraints};

/// Simplex method for linear programming using tensor operations.
///
/// Minimize: c^T * x
/// Subject to:
///   A_ub * x <= b_ub
///   A_eq * x == b_eq
///   lower_bounds <= x <= upper_bounds
pub fn simplex_impl<R, C>(
    client: &C,
    c: &Tensor<R>,
    constraints: &TensorLinearConstraints<R>,
    options: &LinProgOptions,
) -> OptimizeResult<TensorLinProgResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n_orig = c.shape()[0];
    if n_orig == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "simplex: empty objective vector".to_string(),
        });
    }

    // Extract problem data
    let c_data: Vec<f64> = c.to_vec();
    let (lower, upper) = extract_bounds(constraints, n_orig);
    let (a_ub, b_ub) = extract_inequality_data(constraints)?;
    let (a_eq, b_eq) = extract_equality_data(constraints)?;

    // Extend constraints with finite bounds
    let (a_ub_ext, b_ub_ext) = extend_with_bound_constraints(&a_ub, &b_ub, &lower, &upper, n_orig);

    let n_ub = a_ub_ext.len() / n_orig.max(1);
    let n_eq = a_eq.len() / n_orig.max(1);
    let n_constraints = n_ub + n_eq;

    // Handle unconstrained case
    if n_constraints == 0 {
        return solve_unconstrained(client, &c_data, &lower, &upper, n_orig);
    }

    // Problem dimensions for standard form
    let n_slack = n_ub;
    let n_artificial = n_eq + count_negative_rhs(&b_ub_ext);
    let n_total = n_orig + n_slack + n_artificial;
    let n_rows = n_constraints + 1; // constraints + objective
    let n_cols = n_total + 1; // variables + RHS

    // Build initial tableau as tensor
    let (tableau_data, mut basis) = build_tableau_data(
        &c_data,
        &a_ub_ext,
        &b_ub_ext,
        &a_eq,
        &b_eq,
        n_orig,
        n_ub,
        n_eq,
        n_slack,
        n_total,
        n_rows,
        n_cols,
        options.tol,
    );

    let mut tableau = Tensor::<R>::from_slice(&tableau_data, &[n_rows, n_cols], client.device());

    // Simplex iterations
    let mut nit = 0;

    loop {
        if nit >= options.max_iter {
            return make_result(client, &tableau, &basis, n_orig, n_ub, n_cols, &c_data, nit, false, "Maximum iterations reached");
        }

        // Find pivot column (most negative reduced cost in objective row)
        let pivot_col = find_pivot_column_tensor(client, &tableau, n_constraints, n_total, options.tol)?;

        let pivot_col = match pivot_col {
            Some(col) => col,
            None => break, // Optimal solution found
        };

        // Find pivot row (minimum ratio test)
        let pivot_row = find_pivot_row_tensor(client, &tableau, pivot_col, n_constraints, n_cols, options.tol)?;

        let pivot_row = match pivot_row {
            Some(row) => row,
            None => {
                return make_result(client, &tableau, &basis, n_orig, n_ub, n_cols, &c_data, nit, false, "Problem is unbounded");
            }
        };

        // Perform pivot operation using tensor ops
        tableau = pivot_tensor(client, &tableau, pivot_row, pivot_col, n_rows, n_cols)?;
        basis[pivot_row] = pivot_col;
        nit += 1;
    }

    // Check feasibility (artificial variables should be zero)
    if !check_feasibility(&tableau, &basis, n_orig, n_slack, n_cols, options.tol) {
        return make_result(client, &tableau, &basis, n_orig, n_ub, n_cols, &c_data, nit, false, "Problem is infeasible");
    }

    make_result(client, &tableau, &basis, n_orig, n_ub, n_cols, &c_data, nit, true, "Optimal solution found")
}

/// Extract bounds from constraints, defaulting to [0, inf).
fn extract_bounds<R: Runtime>(
    constraints: &TensorLinearConstraints<R>,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let lower = constraints
        .lower_bounds
        .as_ref()
        .map(|t| t.to_vec())
        .unwrap_or_else(|| vec![0.0; n]);
    let upper = constraints
        .upper_bounds
        .as_ref()
        .map(|t| t.to_vec())
        .unwrap_or_else(|| vec![f64::INFINITY; n]);
    (lower, upper)
}

/// Extract inequality constraint data as flat vectors.
fn extract_inequality_data<R: Runtime>(
    constraints: &TensorLinearConstraints<R>,
) -> OptimizeResult<(Vec<f64>, Vec<f64>)> {
    match (&constraints.a_ub, &constraints.b_ub) {
        (Some(a), Some(b)) => Ok((a.to_vec(), b.to_vec())),
        (None, None) => Ok((vec![], vec![])),
        _ => Err(OptimizeError::InvalidInput {
            context: "simplex: A_ub and b_ub must both be provided or both be None".to_string(),
        }),
    }
}

/// Extract equality constraint data as flat vectors.
fn extract_equality_data<R: Runtime>(
    constraints: &TensorLinearConstraints<R>,
) -> OptimizeResult<(Vec<f64>, Vec<f64>)> {
    match (&constraints.a_eq, &constraints.b_eq) {
        (Some(a), Some(b)) => Ok((a.to_vec(), b.to_vec())),
        (None, None) => Ok((vec![], vec![])),
        _ => Err(OptimizeError::InvalidInput {
            context: "simplex: A_eq and b_eq must both be provided or both be None".to_string(),
        }),
    }
}

/// Extend inequality constraints with finite bound constraints.
fn extend_with_bound_constraints(
    a_ub: &[f64],
    b_ub: &[f64],
    lower: &[f64],
    upper: &[f64],
    n_orig: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut a_ext = a_ub.to_vec();
    let mut b_ext = b_ub.to_vec();

    for (i, (&lb, &ub)) in lower.iter().zip(upper.iter()).enumerate() {
        // x_i <= upper_i
        if ub.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = 1.0;
            a_ext.extend(row);
            b_ext.push(ub);
        }
        // -x_i <= -lower_i (for positive lower bounds)
        if lb > 0.0 && lb.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = -1.0;
            a_ext.extend(row);
            b_ext.push(-lb);
        }
    }

    (a_ext, b_ext)
}

fn count_negative_rhs(b: &[f64]) -> usize {
    b.iter().filter(|&&v| v < 0.0).count()
}

/// Build initial tableau data and basis.
#[allow(clippy::too_many_arguments)]
fn build_tableau_data(
    c: &[f64],
    a_ub: &[f64],
    b_ub: &[f64],
    a_eq: &[f64],
    b_eq: &[f64],
    n_orig: usize,
    n_ub: usize,
    n_eq: usize,
    n_slack: usize,
    n_total: usize,
    n_rows: usize,
    n_cols: usize,
    tol: f64,
) -> (Vec<f64>, Vec<usize>) {
    let n_constraints = n_ub + n_eq;
    let big_m = 1e6;

    let mut tableau = vec![0.0; n_rows * n_cols];
    let mut basis = vec![0usize; n_constraints];
    let mut art_idx = n_orig + n_slack;

    // Fill inequality constraints
    for i in 0..n_ub {
        let rhs = b_ub[i];
        let (mult, rhs_val) = if rhs < 0.0 { (-1.0, -rhs) } else { (1.0, rhs) };

        // Copy constraint coefficients
        for j in 0..n_orig {
            tableau[i * n_cols + j] = mult * a_ub[i * n_orig + j];
        }
        tableau[i * n_cols + n_total] = rhs_val; // RHS

        if mult < 0.0 {
            // Need artificial variable
            tableau[i * n_cols + art_idx] = 1.0;
            basis[i] = art_idx;
            art_idx += 1;
        } else {
            // Add slack variable
            tableau[i * n_cols + n_orig + i] = 1.0;
            basis[i] = n_orig + i;
        }
    }

    // Fill equality constraints (always need artificial)
    for i in 0..n_eq {
        let row_idx = n_ub + i;
        let rhs = b_eq[i];
        let (mult, rhs_val) = if rhs < 0.0 { (-1.0, -rhs) } else { (1.0, rhs) };

        for j in 0..n_orig {
            tableau[row_idx * n_cols + j] = mult * a_eq[i * n_orig + j];
        }
        tableau[row_idx * n_cols + n_total] = rhs_val;
        tableau[row_idx * n_cols + art_idx] = 1.0;
        basis[row_idx] = art_idx;
        art_idx += 1;
    }

    // Objective row: c for original vars, big_m for artificial
    let obj_row = n_constraints;
    for (j, &cj) in c.iter().enumerate() {
        tableau[obj_row * n_cols + j] = cj;
    }
    for j in (n_orig + n_slack)..n_total {
        tableau[obj_row * n_cols + j] = big_m;
    }

    // Make objective row canonical by eliminating artificial variable columns
    for i in 0..n_constraints {
        if basis[i] >= n_orig + n_slack {
            let coef = tableau[obj_row * n_cols + basis[i]];
            if coef.abs() > tol {
                for j in 0..n_cols {
                    tableau[obj_row * n_cols + j] -= coef * tableau[i * n_cols + j];
                }
            }
        }
    }

    (tableau, basis)
}

/// Find pivot column using tensor argmin.
/// Returns index of most negative reduced cost, or None if optimal.
fn find_pivot_column_tensor<R, C>(
    client: &C,
    tableau: &Tensor<R>,
    n_constraints: usize,
    n_total: usize,
    tol: f64,
) -> OptimizeResult<Option<usize>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    // Extract objective row (last row, excluding RHS column)
    // First narrow row dimension, then narrow column dimension to exclude RHS
    let obj_row = tableau
        .narrow(0, n_constraints, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_column: narrow row - {}", e),
        })?
        .contiguous()
        .narrow(1, 0, n_total)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_column: narrow cols - {}", e),
        })?
        .contiguous()
        .reshape(&[n_total])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_column: reshape - {}", e),
        })?;

    // Use tensor argmin to find the minimum - stays on device
    let min_idx_tensor = client
        .argmin(&obj_row, 0, false)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_column: argmin - {}", e),
        })?;

    // Extract scalar index (single value, acceptable for control flow)
    let min_idx_data: Vec<i64> = min_idx_tensor.to_vec();
    let min_idx = min_idx_data[0] as usize;

    // Get the minimum value to check if it's negative enough
    let min_val_tensor = obj_row
        .narrow(0, min_idx, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_column: get min val - {}", e),
        })?
        .contiguous();
    let min_val_data: Vec<f64> = min_val_tensor.to_vec();
    let min_val = min_val_data[0];

    if min_val < -tol {
        Ok(Some(min_idx))
    } else {
        Ok(None) // Optimal
    }
}

/// Find pivot row using minimum ratio test with tensor operations.
fn find_pivot_row_tensor<R, C>(
    client: &C,
    tableau: &Tensor<R>,
    pivot_col: usize,
    n_constraints: usize,
    n_cols: usize,
    tol: f64,
) -> OptimizeResult<Option<usize>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    // Extract pivot column (constraint rows only) as 1D tensor
    let col = tableau
        .narrow(0, 0, n_constraints)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: narrow rows - {}", e),
        })?
        .contiguous()
        .narrow(1, pivot_col, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: narrow col - {}", e),
        })?
        .contiguous()
        .reshape(&[n_constraints])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: reshape col - {}", e),
        })?;

    // Extract RHS column as 1D tensor
    let rhs = tableau
        .narrow(0, 0, n_constraints)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: narrow rows rhs - {}", e),
        })?
        .contiguous()
        .narrow(1, n_cols - 1, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: narrow rhs - {}", e),
        })?
        .contiguous()
        .reshape(&[n_constraints])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: reshape rhs - {}", e),
        })?;

    // Create threshold tensor for comparison: col > tol
    let tol_tensor = client
        .fill(&[n_constraints], tol, numr::dtype::DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: create tol tensor - {}", e),
        })?;

    // Create mask: col > tol (positive pivot column elements)
    let mask = client.gt(&col, &tol_tensor).map_err(|e| OptimizeError::NumericalError {
        message: format!("pivot_row: gt comparison - {}", e),
    })?;

    // Compute ratios: rhs / col (will have invalid values where col <= tol)
    let ratios = client.div(&rhs, &col).map_err(|e| OptimizeError::NumericalError {
        message: format!("pivot_row: compute ratios - {}", e),
    })?;

    // Create infinity tensor for invalid ratios
    let infinity = client
        .fill(&[n_constraints], f64::INFINITY, numr::dtype::DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: create infinity tensor - {}", e),
        })?;

    // Apply where_cond: valid_ratios = where(col > tol, ratio, infinity)
    let valid_ratios =
        client
            .where_cond(&mask, &ratios, &infinity)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("pivot_row: where_cond - {}", e),
            })?;

    // Find argmin of valid_ratios
    let min_idx_tensor = client.argmin(&valid_ratios, 0, false).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("pivot_row: argmin - {}", e),
        }
    })?;

    // Extract scalar index for control flow
    let min_idx_data: Vec<i64> = min_idx_tensor.to_vec();
    let min_idx = min_idx_data[0] as usize;

    // Check if the minimum ratio is finite (if all inf, problem is unbounded)
    let min_val_tensor = valid_ratios
        .narrow(0, min_idx, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot_row: get min val - {}", e),
        })?
        .contiguous();
    let min_val_data: Vec<f64> = min_val_tensor.to_vec();
    let min_val = min_val_data[0];

    if min_val.is_finite() {
        Ok(Some(min_idx))
    } else {
        Ok(None) // All ratios are infinite - problem is unbounded
    }
}

/// Perform pivot operation using tensor broadcasting.
fn pivot_tensor<R, C>(
    client: &C,
    tableau: &Tensor<R>,
    pivot_row: usize,
    pivot_col: usize,
    n_rows: usize,
    _n_cols: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Extract pivot element
    let pivot_val = {
        let elem = tableau
            .narrow(0, pivot_row, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("pivot: narrow pivot row - {}", e),
            })?
            .contiguous()
            .narrow(1, pivot_col, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("pivot: narrow pivot col - {}", e),
            })?
            .contiguous();
        let data: Vec<f64> = elem.to_vec();
        data[0]
    };

    if pivot_val.abs() < 1e-15 {
        return Err(OptimizeError::NumericalError {
            message: "pivot: zero pivot element".to_string(),
        });
    }

    // Extract and scale pivot row
    let pivot_row_tensor = tableau
        .narrow(0, pivot_row, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: extract pivot row - {}", e),
        })?
        .contiguous();

    let scaled_pivot_row = client
        .div_scalar(&pivot_row_tensor, pivot_val)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: scale row - {}", e),
        })?;

    // Extract factors column (the column we're eliminating)
    let factors_col = tableau
        .narrow(1, pivot_col, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: extract factors - {}", e),
        })?
        .contiguous();

    // Compute: outer_product = factors_col * scaled_pivot_row
    // Then: new_tableau = tableau - outer_product
    // But we need to NOT modify the pivot row itself

    // Build the update matrix: factors * scaled_pivot_row
    let update = client
        .matmul(&factors_col, &scaled_pivot_row)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: outer product - {}", e),
        })?;

    // Subtract from tableau
    let result = client
        .sub(tableau, &update)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: subtract - {}", e),
        })?;

    // Fix pivot row: it should be scaled_pivot_row, not (tableau[pivot_row] - factor * scaled_pivot_row)
    // The above subtraction zeroed it out. We need to put scaled_pivot_row back.
    // result[pivot_row, :] = scaled_pivot_row

    // Replace pivot row using tensor slicing and concatenation
    let before = if pivot_row > 0 {
        Some(
            result
                .narrow(0, 0, pivot_row)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("pivot: narrow before - {}", e),
                })?
                .contiguous(),
        )
    } else {
        None
    };

    let after = if pivot_row + 1 < n_rows {
        Some(
            result
                .narrow(0, pivot_row + 1, n_rows - pivot_row - 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("pivot: narrow after - {}", e),
                })?
                .contiguous(),
        )
    } else {
        None
    };

    // Concatenate: before | scaled_pivot_row | after
    let mut parts: Vec<&Tensor<R>> = Vec::new();
    if let Some(ref b) = before {
        parts.push(b);
    }
    parts.push(&scaled_pivot_row);
    if let Some(ref a) = after {
        parts.push(a);
    }

    client
        .cat(&parts, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pivot: cat rows - {}", e),
        })
}

/// Check if solution is feasible (no artificial variables in basis with nonzero value).
fn check_feasibility<R: Runtime>(
    tableau: &Tensor<R>,
    basis: &[usize],
    n_orig: usize,
    n_slack: usize,
    n_cols: usize,
    tol: f64,
) -> bool {
    let data: Vec<f64> = tableau.to_vec();
    let n_artificial_start = n_orig + n_slack;

    for (i, &bv) in basis.iter().enumerate() {
        if bv >= n_artificial_start {
            let rhs = data[i * n_cols + n_cols - 1];
            if rhs.abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Build result from final tableau.
#[allow(clippy::too_many_arguments)]
fn make_result<R, C>(
    client: &C,
    tableau: &Tensor<R>,
    basis: &[usize],
    n_orig: usize,
    n_ub: usize,
    n_cols: usize,
    c: &[f64],
    nit: usize,
    success: bool,
    message: &str,
) -> OptimizeResult<TensorLinProgResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let data: Vec<f64> = tableau.to_vec();

    // Extract solution
    let mut x = vec![0.0; n_orig];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < n_orig {
            x[bv] = data[i * n_cols + n_cols - 1].max(0.0);
        }
    }

    // Extract slack variables
    let slack: Vec<f64> = (0..n_ub)
        .map(|i| {
            let slack_var = n_orig + i;
            for (j, &bv) in basis.iter().enumerate() {
                if bv == slack_var {
                    return data[j * n_cols + n_cols - 1].max(0.0);
                }
            }
            0.0
        })
        .collect();

    // Compute objective value
    let fun: f64 = if success {
        x.iter().zip(c.iter()).map(|(&xi, &ci)| xi * ci).sum()
    } else if message.contains("unbounded") {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    let x_tensor = Tensor::<R>::from_slice(&x, &[n_orig], client.device());
    let slack_tensor = if slack.is_empty() {
        Tensor::<R>::from_slice::<f64>(&[], &[0], client.device())
    } else {
        Tensor::<R>::from_slice(&slack, &[slack.len()], client.device())
    };

    Ok(TensorLinProgResult {
        x: x_tensor,
        fun,
        success,
        nit,
        message: message.to_string(),
        slack: slack_tensor,
    })
}

/// Solve unconstrained LP (just optimize at bounds).
fn solve_unconstrained<R, C>(
    client: &C,
    c: &[f64],
    lower: &[f64],
    upper: &[f64],
    n_orig: usize,
) -> OptimizeResult<TensorLinProgResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let mut x = vec![0.0; n_orig];
    let mut fun = 0.0;

    for i in 0..n_orig {
        if c[i] < 0.0 {
            if upper[i].is_infinite() {
                return Err(OptimizeError::InvalidInput {
                    context: "simplex: unbounded problem".to_string(),
                });
            }
            x[i] = upper[i];
        } else {
            x[i] = lower[i];
        }
        fun += c[i] * x[i];
    }

    let x_tensor = Tensor::<R>::from_slice(&x, &[n_orig], client.device());
    let slack_tensor = Tensor::<R>::from_slice::<f64>(&[], &[0], client.device());

    Ok(TensorLinProgResult {
        x: x_tensor,
        fun,
        success: true,
        nit: 0,
        message: "Optimal solution found".to_string(),
        slack: slack_tensor,
    })
}
