//! QP subproblem solver for SLSQP.
//!
//! Solves: min 0.5*d'*B*d + g'd
//!         s.t. A_eq*d = b_eq
//!              A_ineq*d >= b_ineq
//!
//! Uses an active-set method on the KKT system.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Result of QP subproblem.
pub struct QpSubproblemResult<R: Runtime> {
    /// Search direction.
    pub d: Tensor<R>,
    /// Lagrange multipliers for equality constraints.
    pub lambda_eq: Option<Tensor<R>>,
    /// Lagrange multipliers for inequality constraints.
    pub lambda_ineq: Option<Tensor<R>>,
}

/// Solve the QP subproblem arising in SLSQP.
///
/// min  0.5*d'*B*d + g'd
/// s.t. A_eq*d + c_eq = 0       (linearized equality constraints)
///      A_ineq*d + c_ineq >= 0   (linearized inequality constraints)
///
/// B is the BFGS approximation to the Hessian of the Lagrangian.
/// g is the gradient of the objective.
/// A_eq, c_eq are the equality constraint Jacobian and values.
/// A_ineq, c_ineq are the inequality constraint Jacobian and values.
pub fn qp_subproblem_impl<R, C>(
    client: &C,
    b: &Tensor<R>,
    g: &Tensor<R>,
    a_eq: Option<&Tensor<R>>,
    c_eq: Option<&Tensor<R>>,
    a_ineq: Option<&Tensor<R>>,
    c_ineq: Option<&Tensor<R>>,
) -> OptimizeResult<QpSubproblemResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n = g.shape()[0];

    let has_eq = a_eq.is_some() && c_eq.is_some();
    let has_ineq = a_ineq.is_some() && c_ineq.is_some();

    // No constraints: solve B*d = -g directly
    if !has_eq && !has_ineq {
        let neg_g = client
            .mul_scalar(g, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: negate g - {}", e),
            })?;
        let neg_g_col = neg_g
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: reshape - {}", e),
            })?;
        let d_col = LinearAlgebraAlgorithms::solve(client, b, &neg_g_col).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("qp_subproblem: solve B*d=-g - {}", e),
            }
        })?;
        let d = d_col
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: reshape d - {}", e),
            })?;
        return Ok(QpSubproblemResult {
            d,
            lambda_eq: None,
            lambda_ineq: None,
        });
    }

    // Equality-only: solve KKT system
    // [B  A_eq'] [d]      [-g    ]
    // [A_eq  0 ] [lambda] = [-c_eq]
    if has_eq && !has_ineq {
        let a_e = a_eq.ok_or(OptimizeError::InvalidInput {
            context: "qp_subproblem: expected equality constraint Jacobian".to_string(),
        })?;
        let c_e = c_eq.ok_or(OptimizeError::InvalidInput {
            context: "qp_subproblem: expected equality constraint values".to_string(),
        })?;
        let m_eq = a_e.shape()[0];

        let (d, lambda_eq) = solve_equality_qp(client, b, g, a_e, c_e, n, m_eq)?;
        return Ok(QpSubproblemResult {
            d,
            lambda_eq: Some(lambda_eq),
            lambda_ineq: None,
        });
    }

    // General case with inequality constraints: active-set method
    let a_e = a_eq;
    let c_e = c_eq;
    let a_i = a_ineq.ok_or(OptimizeError::InvalidInput {
        context: "qp_subproblem: expected inequality constraint Jacobian".to_string(),
    })?;
    let c_i = c_ineq.ok_or(OptimizeError::InvalidInput {
        context: "qp_subproblem: expected inequality constraint values".to_string(),
    })?;
    let m_ineq = a_i.shape()[0];
    let m_eq = a_e.map_or(0, |a| a.shape()[0]);

    // Start with no active inequality constraints
    let mut active_set: Vec<bool> = vec![false; m_ineq];
    let max_active_iter = 3 * (n + m_ineq);

    let mut d;
    let mut lambda_eq_out = None;
    let mut lambda_ineq_vals = vec![0.0f64; m_ineq];

    for _ in 0..max_active_iter {
        // Build active constraint matrix
        let active_rows: Vec<usize> = active_set
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| i)
            .collect();

        // Combine equality and active inequality constraints
        let (combined_a, combined_c) =
            build_active_constraints(client, a_e, c_e, a_i, c_i, &active_rows, n)?;

        let m_active = m_eq + active_rows.len();

        if m_active == 0 {
            // No active constraints, solve unconstrained
            let neg_g = client
                .mul_scalar(g, -1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("qp_subproblem: negate g - {}", e),
                })?;
            let neg_g_col = neg_g
                .reshape(&[n, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("qp_subproblem: reshape - {}", e),
                })?;
            let d_col = LinearAlgebraAlgorithms::solve(client, b, &neg_g_col).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("qp_subproblem: solve - {}", e),
                }
            })?;
            d = d_col
                .reshape(&[n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("qp_subproblem: reshape d - {}", e),
                })?;
        } else {
            let ca = combined_a.as_ref().ok_or(OptimizeError::InvalidInput {
                context: "qp_subproblem: expected combined constraint matrix".to_string(),
            })?;
            let cc = combined_c.as_ref().ok_or(OptimizeError::InvalidInput {
                context: "qp_subproblem: expected combined constraint values".to_string(),
            })?;
            let (d_sol, lambda) = solve_equality_qp(client, b, g, ca, cc, n, m_active)?;
            d = d_sol;

            // Store equality multipliers
            if m_eq > 0 {
                let eq_lambda =
                    lambda
                        .narrow(0, 0, m_eq)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("qp_subproblem: extract eq lambda - {}", e),
                        })?;
                lambda_eq_out = Some(eq_lambda);
            }

            // Check if any active inequality multiplier is negative -> remove from active set
            let mut removed = false;
            for (idx, &row_i) in active_rows.iter().enumerate() {
                // Extract single multiplier value for this active constraint
                let lam_narrow =
                    lambda
                        .narrow(0, m_eq + idx, 1)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("qp_subproblem: extract multiplier {} - {}", idx, e),
                        })?;
                let lam: f64 = lam_narrow.contiguous().to_vec()[0]; // Single scalar extraction
                lambda_ineq_vals[row_i] = lam;
                if lam < 0.0 {
                    active_set[row_i] = false;
                    removed = true;
                }
            }
            if removed {
                continue;
            }
        }

        // Check inactive inequality constraints for violations
        // c_i + A_i * d >= 0 for all inactive constraints
        let d_col = d
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: reshape d - {}", e),
            })?;
        let a_i_d = client
            .matmul(a_i, &d_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: A_i*d - {}", e),
            })?;
        let a_i_d_flat = a_i_d
            .reshape(&[m_ineq])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: reshape A_i*d - {}", e),
            })?;
        let residual = client
            .add(c_i, &a_i_d_flat)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: c_i + A_i*d - {}", e),
            })?;

        // Find most violated inactive constraint using tensor operations
        let mut worst_violation = 0.0;
        let mut worst_idx = None;
        for (i, is_active) in active_set.iter().enumerate().take(m_ineq) {
            if !*is_active {
                // Extract single residual value for constraint i
                let res_narrow =
                    residual
                        .narrow(0, i, 1)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("qp_subproblem: extract residual {} - {}", i, e),
                        })?;
                let val: f64 = res_narrow.contiguous().to_vec()[0]; // Single scalar extraction
                if val < worst_violation {
                    worst_violation = val;
                    worst_idx = Some(i);
                }
            }
        }

        if let Some(idx) = worst_idx {
            active_set[idx] = true;
        } else {
            // All constraints satisfied -> done
            let lambda_ineq = Tensor::<R>::from_slice(&lambda_ineq_vals, &[m_ineq], d.device());
            return Ok(QpSubproblemResult {
                d,
                lambda_eq: lambda_eq_out,
                lambda_ineq: Some(lambda_ineq),
            });
        }
    }

    // Hit iteration limit, return best solution
    let d_fallback =
        client
            .fill(&[n], 0.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_subproblem: fallback - {}", e),
            })?;
    Ok(QpSubproblemResult {
        d: d_fallback,
        lambda_eq: None,
        lambda_ineq: None,
    })
}

/// Solve equality-constrained QP via KKT system.
///
/// [B  A'] [d]      [-g]
/// [A  0 ] [lambda] = [-c]
fn solve_equality_qp<R, C>(
    client: &C,
    b: &Tensor<R>,
    g: &Tensor<R>,
    a: &Tensor<R>,
    c: &Tensor<R>,
    n: usize,
    m: usize,
) -> OptimizeResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let size = n + m;

    // Build KKT matrix [B, A'; A, 0]
    let a_t = a
        .transpose(0, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: transpose A - {}", e),
        })?;

    let zeros_mm =
        client
            .fill(&[m, m], 0.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("qp_eq: zeros - {}", e),
            })?;

    // Top row: [B, A']
    let top = client
        .cat(&[b, &a_t], 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: cat top - {}", e),
        })?;
    // Bottom row: [A, 0]
    let bottom = client
        .cat(&[a, &zeros_mm], 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: cat bottom - {}", e),
        })?;
    // Full KKT matrix
    let kkt = client
        .cat(&[&top, &bottom], 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: cat kkt - {}", e),
        })?;

    // Build RHS [-g; -c]
    let neg_g = client
        .mul_scalar(g, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: negate g - {}", e),
        })?;
    let neg_c = client
        .mul_scalar(c, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: negate c - {}", e),
        })?;
    let rhs = client
        .cat(&[&neg_g, &neg_c], 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: cat rhs - {}", e),
        })?;
    let rhs_col = rhs
        .reshape(&[size, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: reshape rhs - {}", e),
        })?;

    // Solve KKT system
    let sol = LinearAlgebraAlgorithms::solve(client, &kkt, &rhs_col).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("qp_eq: solve kkt - {}", e),
        }
    })?;
    let sol_flat = sol
        .reshape(&[size])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: reshape sol - {}", e),
        })?;

    // Extract d and lambda
    let d = sol_flat
        .narrow(0, 0, n)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: extract d - {}", e),
        })?
        .contiguous();
    let lambda = sol_flat
        .narrow(0, n, m)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_eq: extract lambda - {}", e),
        })?
        .contiguous();

    Ok((d, lambda))
}

/// Build combined constraint matrix from equality + active inequality constraints.
#[allow(clippy::type_complexity)]
fn build_active_constraints<R, C>(
    client: &C,
    a_eq: Option<&Tensor<R>>,
    c_eq: Option<&Tensor<R>>,
    a_ineq: &Tensor<R>,
    c_ineq: &Tensor<R>,
    active_rows: &[usize],
    _n: usize,
) -> OptimizeResult<(Option<Tensor<R>>, Option<Tensor<R>>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let m_eq = a_eq.map_or(0, |a| a.shape()[0]);
    let m_active = active_rows.len();

    if m_eq == 0 && m_active == 0 {
        return Ok((None, None));
    }

    let mut a_parts: Vec<Tensor<R>> = Vec::new();
    let mut c_parts: Vec<Tensor<R>> = Vec::new();

    if let (Some(ae), Some(ce)) = (a_eq, c_eq) {
        a_parts.push(ae.clone());
        c_parts.push(ce.clone());
    }

    for &row in active_rows {
        let a_row = a_ineq
            .narrow(0, row, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("build_active: narrow A row {} - {}", row, e),
            })?
            .contiguous();
        let c_val = c_ineq
            .narrow(0, row, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("build_active: narrow c row {} - {}", row, e),
            })?
            .contiguous();
        a_parts.push(a_row);
        c_parts.push(c_val);
    }

    let a_refs: Vec<&Tensor<R>> = a_parts.iter().collect();
    let c_refs: Vec<&Tensor<R>> = c_parts.iter().collect();

    let combined_a = client
        .cat(&a_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("build_active: cat A - {}", e),
        })?;
    let combined_c = client
        .cat(&c_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("build_active: cat c - {}", e),
        })?;

    Ok((Some(combined_a), Some(combined_c)))
}
