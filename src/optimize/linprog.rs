//! Linear Programming algorithms.
//!
//! This module provides linear programming solvers including:
//! - `linprog` - Simplex method for linear programming
//! - `milp` - Mixed-integer linear programming via branch-and-bound
//!
//! # Linear Programming Problem
//!
//! Minimize: c^T * x
//! Subject to:
//!   A_ub * x <= b_ub  (inequality constraints)
//!   A_eq * x == b_eq  (equality constraints)
//!   lb <= x <= ub     (bounds)
//!
//! # Example
//!
//! ```ignore
//! use solvr::optimize::linprog::{linprog, LinearConstraints};
//!
//! // Minimize -x - 2y subject to:
//! //   x + y <= 4
//! //   x <= 2
//! //   y <= 3
//! //   x, y >= 0
//! let c = vec![-1.0, -2.0];
//! let constraints = LinearConstraints {
//!     a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
//!     b_ub: Some(vec![4.0, 2.0, 3.0]),
//!     a_eq: None,
//!     b_eq: None,
//!     bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
//! };
//!
//! let result = linprog(&c, &constraints, &LinProgOptions::default())?;
//! // Optimal: x=1, y=3, objective=-7
//! ```

// Indexed loops are clearer for tableau/matrix operations
#![allow(clippy::needless_range_loop)]

use super::error::{OptimizeError, OptimizeResult};

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

/// Options for mixed-integer linear programming.
#[derive(Debug, Clone)]
pub struct MilpOptions {
    /// Maximum number of nodes to explore in branch-and-bound.
    pub max_nodes: usize,
    /// Tolerance for integer feasibility.
    pub int_tol: f64,
    /// Tolerance for optimality gap.
    pub gap_tol: f64,
    /// Base LP solver options.
    pub lp_options: LinProgOptions,
}

impl Default for MilpOptions {
    fn default() -> Self {
        Self {
            max_nodes: 10000,
            int_tol: 1e-6,
            gap_tol: 1e-4,
            lp_options: LinProgOptions::default(),
        }
    }
}

/// Result of mixed-integer linear programming.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Optimal solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value.
    pub fun: f64,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Number of nodes explored.
    pub nodes: usize,
    /// Optimality gap (upper_bound - lower_bound) / |upper_bound|.
    pub gap: f64,
    /// Status message.
    pub message: String,
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

    // Validate constraints
    validate_constraints(n, constraints)?;

    // Convert to standard form and solve
    solve_lp_standard_form(c, constraints, options)
}

/// Validate constraint dimensions.
fn validate_constraints(n: usize, constraints: &LinearConstraints) -> OptimizeResult<()> {
    if let Some(ref a_ub) = constraints.a_ub {
        for (i, row) in a_ub.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: A_ub row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    ),
                });
            }
        }
        if let Some(ref b_ub) = constraints.b_ub {
            if b_ub.len() != a_ub.len() {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: b_ub has {} elements, A_ub has {} rows",
                        b_ub.len(),
                        a_ub.len()
                    ),
                });
            }
        } else {
            return Err(OptimizeError::InvalidInput {
                context: "linprog: A_ub provided but b_ub is missing".to_string(),
            });
        }
    }

    if let Some(ref a_eq) = constraints.a_eq {
        for (i, row) in a_eq.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: A_eq row {} has {} columns, expected {}",
                        i,
                        row.len(),
                        n
                    ),
                });
            }
        }
        if let Some(ref b_eq) = constraints.b_eq {
            if b_eq.len() != a_eq.len() {
                return Err(OptimizeError::InvalidInput {
                    context: format!(
                        "linprog: b_eq has {} elements, A_eq has {} rows",
                        b_eq.len(),
                        a_eq.len()
                    ),
                });
            }
        } else {
            return Err(OptimizeError::InvalidInput {
                context: "linprog: A_eq provided but b_eq is missing".to_string(),
            });
        }
    }

    if let Some(ref bounds) = constraints.bounds {
        if bounds.len() != n {
            return Err(OptimizeError::InvalidInput {
                context: format!(
                    "linprog: bounds has {} elements, expected {}",
                    bounds.len(),
                    n
                ),
            });
        }
        for (i, &(lb, ub)) in bounds.iter().enumerate() {
            if lb > ub {
                return Err(OptimizeError::InvalidInterval {
                    a: lb,
                    b: ub,
                    context: format!("linprog: invalid bounds for variable {}", i),
                });
            }
        }
    }

    Ok(())
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
    // For x_i <= ub_i, add constraint [0,...,1,...,0] * x <= ub_i
    let mut a_ub_extended = constraints.a_ub.clone().unwrap_or_default();
    let mut b_ub_extended = constraints.b_ub.clone().unwrap_or_default();

    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = 1.0;
            a_ub_extended.push(row);
            b_ub_extended.push(ub);
        }
        // Handle finite lower bounds > 0 as -x_i <= -lb_i
        if lb > 0.0 && lb.is_finite() {
            let mut row = vec![0.0; n_orig];
            row[i] = -1.0;
            a_ub_extended.push(row);
            b_ub_extended.push(-lb);
        }
    }

    // Count constraints
    let n_ub = a_ub_extended.len();
    let n_eq = constraints.a_eq.as_ref().map_or(0, |a| a.len());
    let n_constraints = n_ub + n_eq;

    // Handle no constraints case
    if n_constraints == 0 {
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
        return Ok(LinProgResult {
            x,
            fun,
            success: true,
            nit: 0,
            message: "Optimal solution found".to_string(),
            slack: vec![],
        });
    }

    // Build standard form:
    // For inequality Ax <= b: add slack s, so Ax + s = b, s >= 0
    // For equality Ax = b: add artificial variable a for Big-M method
    //
    // Variables: [original vars | slack vars | artificial vars]
    let n_slack = n_ub;
    let n_artificial = n_eq + count_negative_rhs_extended(&b_ub_extended, constraints);
    let n_total = n_orig + n_slack + n_artificial;

    // Big-M for artificial variables
    let big_m = 1e6;

    // Build the tableau
    // Format: [A | I_slack | I_artificial | b]
    //         [c | 0       | M            | 0]
    let mut tableau = vec![vec![0.0; n_total + 1]; n_constraints + 1];
    let mut basis = vec![0usize; n_constraints];
    let mut artificial_in_basis = vec![false; n_constraints];

    let mut art_idx = n_orig + n_slack;

    // Fill inequality constraints (including bound constraints)
    if !a_ub_extended.is_empty() {
        for (i, (row, &rhs)) in a_ub_extended.iter().zip(b_ub_extended.iter()).enumerate() {
            let mut rhs_val = rhs;

            // Handle negative RHS by negating the row
            let mult = if rhs_val < 0.0 {
                rhs_val = -rhs_val;
                -1.0
            } else {
                1.0
            };

            for (j, &val) in row.iter().enumerate() {
                tableau[i][j] = mult * val;
            }
            tableau[i][n_total] = rhs_val;

            if mult < 0.0 {
                // Need artificial variable (constraint became >= after negation)
                tableau[i][art_idx] = 1.0;
                basis[i] = art_idx;
                artificial_in_basis[i] = true;
                art_idx += 1;
            } else {
                // Slack variable
                tableau[i][n_orig + i] = 1.0;
                basis[i] = n_orig + i;
            }
        }
    }

    // Fill equality constraints
    if let (Some(a_eq), Some(b_eq)) = (&constraints.a_eq, &constraints.b_eq) {
        for (i, (row, &rhs)) in a_eq.iter().zip(b_eq.iter()).enumerate() {
            let row_idx = n_ub + i;
            let mut rhs_val = rhs;

            let mult = if rhs_val < 0.0 {
                rhs_val = -rhs_val;
                -1.0
            } else {
                1.0
            };

            for (j, &val) in row.iter().enumerate() {
                tableau[row_idx][j] = mult * val;
            }
            tableau[row_idx][n_total] = rhs_val;

            // Artificial variable for equality constraint
            tableau[row_idx][art_idx] = 1.0;
            basis[row_idx] = art_idx;
            artificial_in_basis[row_idx] = true;
            art_idx += 1;
        }
    }

    // Objective row: minimize c^T x + M * (sum of artificial variables)
    for (j, &cj) in c.iter().enumerate() {
        tableau[n_constraints][j] = cj;
    }
    for j in (n_orig + n_slack)..n_total {
        tableau[n_constraints][j] = big_m;
    }

    // Make objective row canonical by eliminating artificial variables from it
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

        // Find entering variable (most negative reduced cost)
        let mut pivot_col = None;
        let mut min_cost = -options.tol;
        for j in 0..n_total {
            if tableau[n_constraints][j] < min_cost {
                min_cost = tableau[n_constraints][j];
                pivot_col = Some(j);
            }
        }

        // Optimal if no negative reduced cost
        let pivot_col = match pivot_col {
            Some(col) => col,
            None => break,
        };

        // Find leaving variable (minimum ratio test)
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..n_constraints {
            let aij = tableau[i][pivot_col];
            if aij > options.tol {
                let ratio = tableau[i][n_total] / aij;
                if ratio >= -options.tol && ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        // Unbounded if no valid pivot row
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

    // Check if any artificial variable is still in basis with non-zero value
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

    // Calculate objective
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

/// Count rows with negative RHS that need artificial variables.
fn count_negative_rhs_extended(b_ub: &[f64], _constraints: &LinearConstraints) -> usize {
    // Count negative RHS in inequality constraints (including bound constraints)
    b_ub.iter().filter(|&&b| b < 0.0).count()
}

/// Solve a mixed-integer linear programming problem using branch-and-bound.
///
/// # Arguments
///
/// * `c` - Objective function coefficients
/// * `constraints` - Linear constraints
/// * `integrality` - Which variables must be integer (true = integer, false = continuous)
/// * `options` - Solver options
///
/// # Returns
///
/// * `MilpResult` containing optimal integer solution
pub fn milp(
    c: &[f64],
    constraints: &LinearConstraints,
    integrality: &[bool],
    options: &MilpOptions,
) -> OptimizeResult<MilpResult> {
    let n = c.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "milp: empty objective vector".to_string(),
        });
    }

    if integrality.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "milp: integrality has {} elements, expected {}",
                integrality.len(),
                n
            ),
        });
    }

    validate_constraints(n, constraints)?;

    // Branch-and-bound algorithm
    branch_and_bound(c, constraints, integrality, options)
}

/// Node in the branch-and-bound tree.
#[derive(Clone)]
struct BnBNode {
    /// Bounds for this node.
    bounds: Vec<(f64, f64)>,
    /// Lower bound on objective (from LP relaxation).
    lower_bound: f64,
}

/// Branch-and-bound solver for MILP.
fn branch_and_bound(
    c: &[f64],
    constraints: &LinearConstraints,
    integrality: &[bool],
    options: &MilpOptions,
) -> OptimizeResult<MilpResult> {
    let n = c.len();

    // Get initial bounds
    let base_bounds: Vec<(f64, f64)> = constraints
        .bounds
        .clone()
        .unwrap_or_else(|| vec![(0.0, f64::INFINITY); n]);

    // Initialize with root node
    let mut stack: Vec<BnBNode> = vec![BnBNode {
        bounds: base_bounds.clone(),
        lower_bound: f64::NEG_INFINITY,
    }];

    let mut best_solution: Option<Vec<f64>> = None;
    let mut best_objective = f64::INFINITY;
    let mut nodes_explored = 0;

    while let Some(node) = stack.pop() {
        nodes_explored += 1;

        if nodes_explored > options.max_nodes {
            break;
        }

        // Prune if lower bound exceeds best known solution
        if node.lower_bound >= best_objective - options.gap_tol {
            continue;
        }

        // Solve LP relaxation with current bounds
        let node_constraints = LinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            bounds: Some(node.bounds.clone()),
        };

        let lp_result = match linprog(c, &node_constraints, &options.lp_options) {
            Ok(r) => r,
            Err(_) => continue,
        };

        if !lp_result.success {
            continue;
        }

        // Prune if LP relaxation is worse than best known
        if lp_result.fun >= best_objective - options.gap_tol {
            continue;
        }

        // Check if solution is integer feasible
        let mut is_integer_feasible = true;
        let mut branch_var = None;
        let mut max_fractionality = 0.0;

        for (i, (&is_int, &xi)) in integrality.iter().zip(lp_result.x.iter()).enumerate() {
            if is_int {
                let frac = xi - xi.floor();
                let fractionality = frac.min(1.0 - frac);
                if fractionality > options.int_tol {
                    is_integer_feasible = false;
                    if fractionality > max_fractionality {
                        max_fractionality = fractionality;
                        branch_var = Some(i);
                    }
                }
            }
        }

        if is_integer_feasible {
            if lp_result.fun < best_objective {
                best_objective = lp_result.fun;
                best_solution = Some(lp_result.x.clone());
            }
            continue;
        }

        // Branch on the most fractional variable
        if let Some(var) = branch_var {
            let xi = lp_result.x[var];
            let floor_val = xi.floor();
            let ceil_val = xi.ceil();

            // Create child node with x[var] <= floor
            let mut left_bounds = node.bounds.clone();
            left_bounds[var].1 = left_bounds[var].1.min(floor_val);
            if left_bounds[var].0 <= left_bounds[var].1 {
                stack.push(BnBNode {
                    bounds: left_bounds,
                    lower_bound: lp_result.fun,
                });
            }

            // Create child node with x[var] >= ceil
            let mut right_bounds = node.bounds.clone();
            right_bounds[var].0 = right_bounds[var].0.max(ceil_val);
            if right_bounds[var].0 <= right_bounds[var].1 {
                stack.push(BnBNode {
                    bounds: right_bounds,
                    lower_bound: lp_result.fun,
                });
            }
        }
    }

    match best_solution {
        Some(x) => {
            // Round integer variables
            let x_rounded: Vec<f64> = x
                .iter()
                .zip(integrality.iter())
                .map(|(&xi, &is_int)| if is_int { xi.round() } else { xi })
                .collect();

            let fun: f64 = x_rounded
                .iter()
                .zip(c.iter())
                .map(|(&xi, &ci)| xi * ci)
                .sum();

            let gap = if best_objective.abs() > 1e-10 {
                (best_objective - fun).abs() / best_objective.abs()
            } else {
                0.0
            };

            Ok(MilpResult {
                x: x_rounded,
                fun,
                success: true,
                nodes: nodes_explored,
                gap,
                message: "Optimal solution found".to_string(),
            })
        }
        None => Ok(MilpResult {
            x: vec![0.0; n],
            fun: f64::INFINITY,
            success: false,
            nodes: nodes_explored,
            gap: f64::INFINITY,
            message: if nodes_explored >= options.max_nodes {
                "Maximum nodes reached without finding feasible solution".to_string()
            } else {
                "No feasible integer solution found".to_string()
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linprog_simple() {
        // Minimize -x - 2y
        // Subject to: x + y <= 4, x <= 2, y <= 3, x,y >= 0
        // Optimal: x=1, y=3, obj=-7
        let c = vec![-1.0, -2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 2.0, 3.0]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");

        assert!(result.success, "LP should succeed: {}", result.message);
        assert!(
            (result.fun - (-7.0)).abs() < 0.1,
            "Expected obj=-7, got {}",
            result.fun
        );
    }

    #[test]
    fn test_linprog_unbounded_returns_error() {
        // Minimize -x with no upper bound
        let c = vec![-1.0];
        let constraints = LinearConstraints {
            a_ub: None,
            b_ub: None,
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY)]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default());
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[test]
    fn test_linprog_with_equality() {
        // Minimize x + y
        // Subject to: x + y = 2, x,y >= 0
        // Optimal: any point on x + y = 2 with min cost, so x=0,y=2 or x=2,y=0
        let c = vec![1.0, 1.0];
        let constraints = LinearConstraints {
            a_ub: None,
            b_ub: None,
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![2.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");

        assert!(result.success, "LP should succeed: {}", result.message);
        assert!(
            (result.fun - 2.0).abs() < 0.1,
            "Expected obj=2, got {}",
            result.fun
        );
    }

    #[test]
    fn test_linprog_canonical_form() {
        // Minimize 2x + 3y
        // Subject to: x + y >= 1, 2x + y >= 2, x,y >= 0
        // Convert >= to <=: -x - y <= -1, -2x - y <= -2
        // Optimal: x=1, y=0, obj=2
        let c = vec![2.0, 3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![-1.0, -1.0], vec![-2.0, -1.0]]),
            b_ub: Some(vec![-1.0, -2.0]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");

        assert!(result.success, "LP should succeed: {}", result.message);
        assert!(
            (result.fun - 2.0).abs() < 0.5,
            "Expected obj~2, got {}",
            result.fun
        );
    }

    #[test]
    fn test_linprog_empty_objective() {
        let result = linprog(
            &[],
            &LinearConstraints::default(),
            &LinProgOptions::default(),
        );
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0]]), // Wrong dimension
            b_ub: Some(vec![1.0]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_milp_simple() {
        // Minimize -x - 2y
        // Subject to: x + y <= 4, x <= 2.5, x,y >= 0, x,y integer
        let c = vec![-1.0, -2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0]]),
            b_ub: Some(vec![4.0, 2.5]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };
        let integrality = vec![true, true];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success, "MILP should succeed: {}", result.message);
        assert!(
            result.fun <= -6.0 + 0.1,
            "Expected obj<=-6, got {}",
            result.fun
        );

        // Check integrality
        for (i, &is_int) in integrality.iter().enumerate() {
            if is_int {
                assert!(
                    (result.x[i] - result.x[i].round()).abs() < 0.01,
                    "x[{}]={} should be integer",
                    i,
                    result.x[i]
                );
            }
        }
    }

    #[test]
    fn test_milp_binary() {
        // Binary knapsack: maximize 3x + 4y subject to 2x + 3y <= 5, x,y in {0,1}
        // Equivalent to: minimize -3x - 4y
        // Optimal: x=1, y=1, obj=-7
        let c = vec![-3.0, -4.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![2.0, 3.0]]),
            b_ub: Some(vec![5.0]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, 1.0), (0.0, 1.0)]),
        };
        let integrality = vec![true, true];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success, "MILP should succeed: {}", result.message);
        assert!(
            (result.fun - (-7.0)).abs() < 0.1,
            "Expected obj=-7, got {}",
            result.fun
        );
    }

    #[test]
    fn test_milp_mixed() {
        // Mixed-integer: one integer, one continuous
        // Minimize -x - y, x + y <= 2.5, x integer, y continuous, x,y >= 0
        let c = vec![-1.0, -1.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![2.5]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };
        let integrality = vec![true, false];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success, "MILP should succeed: {}", result.message);
        assert!(
            result.fun <= -2.5 + 0.1,
            "Expected obj<=-2.5, got {}",
            result.fun
        );
        assert!(
            (result.x[0] - result.x[0].round()).abs() < 0.01,
            "x should be integer"
        );
    }

    #[test]
    fn test_milp_empty_integrality() {
        let c = vec![1.0, 2.0];
        let result = milp(
            &c,
            &LinearConstraints::default(),
            &[],
            &MilpOptions::default(),
        );
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_linprog_3d() {
        // Minimize x + 2y + 3z
        // Subject to: x + y + z >= 3 (i.e., -x - y - z <= -3)
        //             x, y, z >= 0
        // Optimal: x=3, y=0, z=0, obj=3
        let c = vec![1.0, 2.0, 3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![-1.0, -1.0, -1.0]]),
            b_ub: Some(vec![-3.0]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");

        assert!(result.success, "LP should succeed: {}", result.message);
        assert!(
            (result.fun - 3.0).abs() < 0.5,
            "Expected obj~3, got {}",
            result.fun
        );
    }

    #[test]
    fn test_linprog_maximize() {
        // Maximize 2x + 3y is equivalent to minimize -2x - 3y
        // Subject to: x + y <= 5, x <= 3, y <= 4, x,y >= 0
        // Optimal: x=1, y=4, max_obj=14 (min_obj=-14)
        let c = vec![-2.0, -3.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![5.0, 3.0, 4.0]),
            a_eq: None,
            b_eq: None,
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");

        assert!(result.success, "LP should succeed: {}", result.message);
        assert!(
            (result.fun - (-14.0)).abs() < 0.5,
            "Expected obj~-14, got {}",
            result.fun
        );
    }
}
