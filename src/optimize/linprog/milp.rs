//! Mixed-integer linear programming via branch-and-bound.

use super::{linprog, validate_constraints, LinearConstraints, LinProgOptions};
use crate::optimize::error::{OptimizeError, OptimizeResult};

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
    branch_and_bound(c, constraints, integrality, options)
}

/// Node in the branch-and-bound tree.
#[derive(Clone)]
struct BnBNode {
    bounds: Vec<(f64, f64)>,
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

    let base_bounds: Vec<(f64, f64)> = constraints
        .bounds
        .clone()
        .unwrap_or_else(|| vec![(0.0, f64::INFINITY); n]);

    let mut stack: Vec<BnBNode> = vec![BnBNode {
        bounds: base_bounds,
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

        if node.lower_bound >= best_objective - options.gap_tol {
            continue;
        }

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

        if lp_result.fun >= best_objective - options.gap_tol {
            continue;
        }

        // Check integer feasibility and find branching variable
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

            // Left child: x[var] <= floor
            let mut left_bounds = node.bounds.clone();
            left_bounds[var].1 = left_bounds[var].1.min(floor_val);
            if left_bounds[var].0 <= left_bounds[var].1 {
                stack.push(BnBNode {
                    bounds: left_bounds,
                    lower_bound: lp_result.fun,
                });
            }

            // Right child: x[var] >= ceil
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
    fn test_milp_simple() {
        let c = vec![-1.0, -2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0]]),
            b_ub: Some(vec![4.0, 2.5]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };
        let integrality = vec![true, true];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success);
        assert!(result.fun <= -6.0 + 0.1);

        for (i, &is_int) in integrality.iter().enumerate() {
            if is_int {
                assert!((result.x[i] - result.x[i].round()).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_milp_binary() {
        let c = vec![-3.0, -4.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![2.0, 3.0]]),
            b_ub: Some(vec![5.0]),
            bounds: Some(vec![(0.0, 1.0), (0.0, 1.0)]),
            ..Default::default()
        };
        let integrality = vec![true, true];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success);
        assert!((result.fun - (-7.0)).abs() < 0.1);
    }

    #[test]
    fn test_milp_mixed() {
        let c = vec![-1.0, -1.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![2.5]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };
        let integrality = vec![true, false];

        let result =
            milp(&c, &constraints, &integrality, &MilpOptions::default()).expect("milp failed");

        assert!(result.success);
        assert!(result.fun <= -2.5 + 0.1);
        assert!((result.x[0] - result.x[0].round()).abs() < 0.01);
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
        assert!(matches!(
            result,
            Err(crate::optimize::error::OptimizeError::InvalidInput { .. })
        ));
    }
}
