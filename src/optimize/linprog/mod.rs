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

mod milp;
mod simplex;

pub use milp::{MilpOptions, MilpResult, milp};
pub use simplex::{LinProgOptions, LinProgResult, LinearConstraints, linprog};

use super::error::{OptimizeError, OptimizeResult};

/// Validate constraint dimensions.
pub(crate) fn validate_constraints(
    n: usize,
    constraints: &LinearConstraints,
) -> OptimizeResult<()> {
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
