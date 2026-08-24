//! Tensor-based mixed-integer linear programming via branch-and-bound.
use crate::DType;

use super::{TensorLinProgResult, TensorLinearConstraints, simplex_impl};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::linprog::LinProgOptions;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Result from tensor-based MILP.
#[derive(Debug, Clone)]
pub struct TensorMilpResult<R: Runtime<DType = DType>> {
    /// Optimal solution vector.
    pub x: Tensor<R>,
    /// Optimal objective value.
    pub fun: f64,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Number of nodes explored.
    pub nodes: usize,
    /// Optimality gap.
    pub gap: f64,
    /// Status message.
    pub message: String,
}

/// Options for mixed-integer linear programming.
#[derive(Debug, Clone)]
pub struct MilpOptionsInternal {
    /// Maximum number of nodes to explore in branch-and-bound.
    pub max_nodes: usize,
    /// Tolerance for integer feasibility.
    pub int_tol: f64,
    /// Tolerance for optimality gap.
    pub gap_tol: f64,
    /// Base LP solver options.
    pub lp_options: LinProgOptions,
}

/// Generic MILP implementation using branch-and-bound.
pub fn milp_impl<R, C>(
    client: &C,
    c: &Tensor<R>,
    constraints: &TensorLinearConstraints<R>,
    integrality: &Tensor<R>,
    options: &MilpOptionsInternal,
) -> OptimizeResult<TensorMilpResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n = c.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "milp: empty objective vector".to_string(),
        });
    }

    let int_data: Vec<f64> = integrality.to_vec();
    if int_data.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "milp: integrality has {} elements, expected {}",
                int_data.len(),
                n
            ),
        });
    }

    // Get base bounds
    let (base_lower, base_upper) = match (&constraints.lower_bounds, &constraints.upper_bounds) {
        (Some(lb), Some(ub)) => (lb.to_vec(), ub.to_vec()),
        _ => (vec![0.0; n], vec![f64::INFINITY; n]),
    };

    // Branch-and-bound state
    struct BnBNode {
        lower: Vec<f64>,
        upper: Vec<f64>,
        lower_bound: f64,
    }

    let mut stack: Vec<BnBNode> = vec![BnBNode {
        lower: base_lower,
        upper: base_upper,
        lower_bound: f64::NEG_INFINITY,
    }];

    let mut best_solution: Option<Vec<f64>> = None;
    let mut best_objective = f64::INFINITY;
    let mut nodes_explored = 0;

    let device = c.device();

    while let Some(node) = stack.pop() {
        nodes_explored += 1;

        if nodes_explored > options.max_nodes {
            break;
        }

        if node.lower_bound >= best_objective - options.gap_tol {
            continue;
        }

        // Create constraints for this node
        let node_lower = Tensor::<R>::from_slice(&node.lower, &[n], device)?;
        let node_upper = Tensor::<R>::from_slice(&node.upper, &[n], device)?;

        let node_constraints = TensorLinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            lower_bounds: Some(node_lower),
            upper_bounds: Some(node_upper),
        };

        // Solve LP relaxation
        let lp_result: TensorLinProgResult<R> =
            match simplex_impl(client, c, &node_constraints, &options.lp_options) {
                Ok(r) => r,
                Err(_) => continue,
            };

        if !lp_result.success {
            continue;
        }

        if lp_result.fun >= best_objective - options.gap_tol {
            continue;
        }

        // Extract solution and check integer feasibility
        let x_data: Vec<f64> = lp_result.x.to_vec();

        let mut is_integer_feasible = true;
        let mut branch_var = None;
        let mut max_fractionality = 0.0;

        for (i, (&is_int, &xi)) in int_data.iter().zip(x_data.iter()).enumerate() {
            if is_int > 0.5 {
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
                best_solution = Some(x_data);
            }
            continue;
        }

        // Branch on the most fractional variable
        if let Some(var) = branch_var {
            let xi = x_data[var];
            let floor_val = xi.floor();
            let ceil_val = xi.ceil();

            // Left child: x[var] <= floor
            let mut left_upper = node.upper.clone();
            left_upper[var] = left_upper[var].min(floor_val);
            if node.lower[var] <= left_upper[var] {
                stack.push(BnBNode {
                    lower: node.lower.clone(),
                    upper: left_upper,
                    lower_bound: lp_result.fun,
                });
            }

            // Right child: x[var] >= ceil
            let mut right_lower = node.lower.clone();
            right_lower[var] = right_lower[var].max(ceil_val);
            if right_lower[var] <= node.upper[var] {
                stack.push(BnBNode {
                    lower: right_lower,
                    upper: node.upper,
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
                .zip(int_data.iter())
                .map(|(&xi, &is_int)| if is_int > 0.5 { xi.round() } else { xi })
                .collect();

            // Compute objective
            let c_data: Vec<f64> = c.to_vec();
            let fun: f64 = x_rounded
                .iter()
                .zip(c_data.iter())
                .map(|(&xi, &ci)| xi * ci)
                .sum();

            let gap = if best_objective.abs() > 1e-10 {
                (best_objective - fun).abs() / best_objective.abs()
            } else {
                0.0
            };

            let x_tensor = Tensor::<R>::from_slice(&x_rounded, &[n], device)?;

            Ok(TensorMilpResult {
                x: x_tensor,
                fun,
                success: true,
                nodes: nodes_explored,
                gap,
                message: "Optimal solution found".to_string(),
            })
        }
        None => {
            let zeros = vec![0.0; n];
            let x_tensor = Tensor::<R>::from_slice(&zeros, &[n], device)?;

            Ok(TensorMilpResult {
                x: x_tensor,
                fun: f64::INFINITY,
                success: false,
                nodes: nodes_explored,
                gap: f64::INFINITY,
                message: if nodes_explored >= options.max_nodes {
                    "Maximum nodes reached without finding feasible solution".to_string()
                } else {
                    "No feasible integer solution found".to_string()
                },
            })
        }
    }
}
