//! Simplex method for linear programming.

#![allow(clippy::needless_range_loop)]

use super::validate_constraints;
use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Options for linear programming solvers.
#[derive(Debug, Clone)]
pub struct LinProgOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for optimality.
    pub tol: f64,
    /// Whether to presolve (remove redundant constraints).
    pub presolve: bool,
}

impl Default for LinProgOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-9,
            presolve: true,
        }
    }
}

/// Linear constraints for LP problems.
#[derive(Debug, Clone, Default)]
pub struct LinearConstraints {
    /// Inequality constraint matrix (A_ub * x <= b_ub).
    pub a_ub: Option<Vec<Vec<f64>>>,
    /// Inequality constraint bounds.
    pub b_ub: Option<Vec<f64>>,
    /// Equality constraint matrix (A_eq * x == b_eq).
    pub a_eq: Option<Vec<Vec<f64>>>,
    /// Equality constraint bounds.
    pub b_eq: Option<Vec<f64>>,
    /// Variable bounds as (lower, upper) pairs. Use f64::NEG_INFINITY/INFINITY for unbounded.
    pub bounds: Option<Vec<(f64, f64)>>,
}

/// Result of linear programming optimization.
#[derive(Debug, Clone)]
pub struct LinProgResult {
    /// Optimal solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value.
    pub fun: f64,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Number of iterations performed.
    pub nit: usize,
    /// Status message.
    pub message: String,
    /// Slack variables for inequality constraints.
    pub slack: Vec<f64>,
}

/// Solve a linear programming problem using the Simplex method.
///
/// Minimize: c^T * x
/// Subject to:
///   A_ub * x <= b_ub
///   A_eq * x == b_eq
///   bounds.0 <= x <= bounds.1
///
/// # Arguments
///
/// * `c` - Objective function coefficients (minimize c^T * x)
/// * `constraints` - Linear constraints (inequalities, equalities, bounds)
/// * `options` - Solver options
///
/// # Returns
///
/// * `LinProgResult` containing optimal solution and objective value
pub fn linprog(
    c: &[f64],
    constraints: &LinearConstraints,
    options: &LinProgOptions,
) -> OptimizeResult<LinProgResult> {
    let n = c.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "linprog: empty objective vector".to_string(),
        });
    }

    validate_constraints(n, constraints)?;
    solve_lp_standard_form(c, constraints, options)
}

/// Solve LP in standard form using the revised simplex method with Big-M.
fn solve_lp_standard_form(
    c: &[f64],
    constraints: &LinearConstraints,
    options: &LinProgOptions,
) -> OptimizeResult<LinProgResult> {
    let n_orig = c.len();

    // Get bounds (default to [0, inf) for each variable)
    let bounds: Vec<(f64, f64)> = constraints
        .bounds
        .clone()
        .unwrap_or_else(|| vec![(0.0, f64::INFINITY); n_orig]);

    // Convert finite upper bounds to explicit inequality constraints
    let mut a_ub_extended = constraints.a_ub.clone().unwrap_or_default();
    let mut b_ub_extended = constraints.b_ub.clone().unwrap_or_default();

    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = 1.0;
            a_ub_extended.push(row);
            b_ub_extended.push(ub);
        }
        if lb > 0.0 && lb.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = -1.0;
            a_ub_extended.push(row);
            b_ub_extended.push(-lb);
        }
    }

    let n_ub = a_ub_extended.len();
    let n_eq = constraints.a_eq.as_ref().map_or(0, |a| a.len());
    let n_constraints = n_ub + n_eq;

    // Handle no constraints case
    if n_constraints == 0 {
        return solve_unconstrained(c, &bounds, n_orig);
    }

    // Build standard form with slack and artificial variables
    let n_slack = n_ub;
    let n_artificial = n_eq + count_negative_rhs(&b_ub_extended);
    let n_total = n_orig + n_slack + n_artificial;
    let big_m = 1e6;

    let mut tableau = vec![vec![0.0; n_total + 1]; n_constraints + 1];
    let mut basis = vec![0usize; n_constraints];
    let mut artificial_in_basis = vec![false; n_constraints];
    let mut art_idx = n_orig + n_slack;

    // Fill inequality constraints
    for (i, (row, &rhs)) in a_ub_extended.iter().zip(b_ub_extended.iter()).enumerate() {
        let (mult, rhs_val) = if rhs < 0.0 { (-1.0, -rhs) } else { (1.0, rhs) };

        for (j, &val) in row.iter().enumerate() {
            tableau[i][j] = mult * val;
        }
        tableau[i][n_total] = rhs_val;

        if mult < 0.0 {
            tableau[i][art_idx] = 1.0;
            basis[i] = art_idx;
            artificial_in_basis[i] = true;
            art_idx += 1;
        } else {
            tableau[i][n_orig + i] = 1.0;
            basis[i] = n_orig + i;
        }
    }

    // Fill equality constraints
    if let (Some(a_eq), Some(b_eq)) = (&constraints.a_eq, &constraints.b_eq) {
        for (i, (row, &rhs)) in a_eq.iter().zip(b_eq.iter()).enumerate() {
            let row_idx = n_ub + i;
            let (mult, rhs_val) = if rhs < 0.0 { (-1.0, -rhs) } else { (1.0, rhs) };

            for (j, &val) in row.iter().enumerate() {
                tableau[row_idx][j] = mult * val;
            }
            tableau[row_idx][n_total] = rhs_val;
            tableau[row_idx][art_idx] = 1.0;
            basis[row_idx] = art_idx;
            artificial_in_basis[row_idx] = true;
            art_idx += 1;
        }
    }

    // Objective row
    for (j, &cj) in c.iter().enumerate() {
        tableau[n_constraints][j] = cj;
    }
    for j in (n_orig + n_slack)..n_total {
        tableau[n_constraints][j] = big_m;
    }

    // Make objective row canonical
    for (i, &is_art) in artificial_in_basis.iter().enumerate() {
        if is_art {
            let basic_var = basis[i];
            let coef = tableau[n_constraints][basic_var];
            if coef.abs() > options.tol {
                for j in 0..=n_total {
                    tableau[n_constraints][j] -= coef * tableau[i][j];
                }
            }
        }
    }

    // Simplex iterations
    let mut nit = 0;
    loop {
        if nit >= options.max_iter {
            return Ok(LinProgResult {
                x: vec![0.0; n_orig],
                fun: f64::INFINITY,
                success: false,
                nit,
                message: "Maximum iterations reached".to_string(),
                slack: vec![],
            });
        }

        // Find entering variable
        let pivot_col = (0..n_total)
            .filter(|&j| tableau[n_constraints][j] < -options.tol)
            .min_by(|&a, &b| {
                tableau[n_constraints][a]
                    .partial_cmp(&tableau[n_constraints][b])
                    .unwrap()
            });

        let pivot_col = match pivot_col {
            Some(col) => col,
            None => break,
        };

        // Find leaving variable
        let pivot_row = (0..n_constraints)
            .filter(|&i| tableau[i][pivot_col] > options.tol)
            .min_by(|&a, &b| {
                let ratio_a = tableau[a][n_total] / tableau[a][pivot_col];
                let ratio_b = tableau[b][n_total] / tableau[b][pivot_col];
                ratio_a.partial_cmp(&ratio_b).unwrap()
            });

        let pivot_row = match pivot_row {
            Some(row) => row,
            None => {
                return Ok(LinProgResult {
                    x: vec![0.0; n_orig],
                    fun: f64::NEG_INFINITY,
                    success: false,
                    nit,
                    message: "Problem is unbounded".to_string(),
                    slack: vec![],
                });
            }
        };

        // Pivot operation
        let pivot_val = tableau[pivot_row][pivot_col];
        for j in 0..=n_total {
            tableau[pivot_row][j] /= pivot_val;
        }
        for i in 0..=n_constraints {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                for j in 0..=n_total {
                    tableau[i][j] -= factor * tableau[pivot_row][j];
                }
            }
        }

        basis[pivot_row] = pivot_col;
        nit += 1;
    }

    // Check feasibility
    for (i, &bv) in basis.iter().enumerate() {
        if bv >= n_orig + n_slack && tableau[i][n_total].abs() > options.tol {
            return Ok(LinProgResult {
                x: vec![0.0; n_orig],
                fun: f64::INFINITY,
                success: false,
                nit,
                message: "Problem is infeasible".to_string(),
                slack: vec![],
            });
        }
    }

    // Extract solution
    let mut x = vec![0.0; n_orig];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < n_orig {
            x[bv] = tableau[i][n_total];
        }
    }

    // Apply bounds correction
    for i in 0..n_orig {
        x[i] = x[i].max(bounds[i].0);
        if bounds[i].1.is_finite() {
            x[i] = x[i].min(bounds[i].1);
        }
    }

    // Calculate slack
    let slack: Vec<f64> = (0..n_ub)
        .map(|i| {
            let slack_var = n_orig + i;
            for (j, &bv) in basis.iter().enumerate() {
                if bv == slack_var {
                    return tableau[j][n_total].max(0.0);
                }
            }
            0.0
        })
        .collect();

    let fun: f64 = x.iter().zip(c.iter()).map(|(&xi, &ci)| xi * ci).sum();

    Ok(LinProgResult {
        x,
        fun,
        success: true,
        nit,
        message: "Optimal solution found".to_string(),
        slack,
    })
}

/// Solve unconstrained LP.
fn solve_unconstrained(
    c: &[f64],
    bounds: &[(f64, f64)],
    n_orig: usize,
) -> OptimizeResult<LinProgResult> {
    let mut x = vec![0.0; n_orig];
    let mut fun = 0.0;
    for i in 0..n_orig {
        if c[i] < 0.0 {
            if bounds[i].1.is_infinite() {
                return Err(OptimizeError::InvalidInput {
                    context: "linprog: unbounded problem".to_string(),
                });
            }
            x[i] = bounds[i].1;
        } else {
            x[i] = bounds[i].0;
        }
        fun += c[i] * x[i];
    }
    Ok(LinProgResult {
        x,
        fun,
        success: true,
        nit: 0,
        message: "Optimal solution found".to_string(),
        slack: vec![],
    })
}

/// Count rows with negative RHS.
fn count_negative_rhs(b_ub: &[f64]) -> usize {
    b_ub.iter().filter(|&&b| b < 0.0).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linprog_simple() {
        let c = vec![-1.0, -2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 2.0, 3.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - (-7.0)).abs() < 0.1);
    }

    #[test]
    fn test_linprog_unbounded() {
        let c = vec![-1.0];
        let constraints = LinearConstraints {
            bounds: Some(vec![(0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default());
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[test]
    fn test_linprog_with_equality() {
        let c = vec![1.0, 1.0];
        let constraints = LinearConstraints {
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![2.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_linprog_canonical_form() {
        let c = vec![2.0, 3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![-1.0, -1.0], vec![-2.0, -1.0]]),
            b_ub: Some(vec![-1.0, -2.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_linprog_empty_objective() {
        let result = linprog(
            &[],
            &LinearConstraints::default(),
            &LinProgOptions::default(),
        );
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0]]),
            b_ub: Some(vec![1.0]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default());
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_linprog_3d() {
        let c = vec![1.0, 2.0, 3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![-1.0, -1.0, -1.0]]),
            b_ub: Some(vec![-3.0]),
            bounds: Some(vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_linprog_maximize() {
        let c = vec![-2.0, -3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![5.0, 3.0, 4.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - (-14.0)).abs() < 0.5);
    }
}
