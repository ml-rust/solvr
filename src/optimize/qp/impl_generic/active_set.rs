//! Active set method for quadratic programming.
//!
//! Solves: min 0.5*x'*Q*x + c'*x s.t. linear constraints.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::impl_generic::utils::{tensor_dot, tensor_norm};
use crate::optimize::qp::traits::{QpOptions, QpResult};

/// Active set QP solver.
#[allow(clippy::too_many_arguments)]
pub fn active_set_qp_impl<R, C>(
    client: &C,
    q: &Tensor<R>,
    c_vec: &Tensor<R>,
    a_eq: Option<&Tensor<R>>,
    b_eq: Option<&Tensor<R>>,
    a_ineq: Option<&Tensor<R>>,
    b_ineq: Option<&Tensor<R>>,
    options: &QpOptions,
) -> OptimizeResult<QpResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + LinearAlgebraAlgorithms<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + RuntimeClient<R>,
{
    let n = c_vec.shape()[0];
    let m_eq = a_eq.map_or(0, |a| a.shape()[0]);
    let m_ineq = a_ineq.map_or(0, |a| a.shape()[0]);

    // Find initial feasible point (simple: start at origin or solve feasibility)
    let mut x = client
        .fill(&[n], 0.0, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("active_set: initial x - {}", e),
        })?;

    // If we have equality constraints, find a feasible point via least squares
    if let (Some(ae), Some(be)) = (a_eq, b_eq) {
        let be_col = be
            .reshape(&[m_eq, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: reshape b_eq - {}", e),
            })?;
        x = LinearAlgebraAlgorithms::lstsq(client, ae, &be_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: initial feasible - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: reshape x - {}", e),
            })?;
    }

    let mut active_set: Vec<bool> = vec![false; m_ineq];

    for iter in 0..options.max_iter {
        // Compute gradient: g = Q*x + c
        let x_col = x
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: reshape x - {}", e),
            })?;
        let qx = client
            .matmul(q, &x_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: Q*x - {}", e),
            })?;
        let qx_flat = qx
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: reshape Qx - {}", e),
            })?;
        let grad = client
            .add(&qx_flat, c_vec)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: grad = Qx+c - {}", e),
            })?;

        // Build working constraint set (equality + active inequality)
        let active_rows: Vec<usize> = active_set
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| i)
            .collect();

        let m_working = m_eq + active_rows.len();

        if m_working == 0 {
            // No constraints active: check for unconstrained minimum
            let grad_norm =
                tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: grad norm - {}", e),
                })?;

            if grad_norm < options.tol {
                let fun = compute_qp_objective(client, q, c_vec, &x, n)?;
                return Ok(QpResult {
                    x,
                    fun,
                    iterations: iter + 1,
                    converged: true,
                    dual_eq: None,
                    dual_ineq: None,
                });
            }

            // Solve Q*d = -grad
            let neg_grad =
                client
                    .mul_scalar(&grad, -1.0)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("active_set: negate grad - {}", e),
                    })?;
            let neg_grad_col =
                neg_grad
                    .reshape(&[n, 1])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("active_set: reshape neg_grad - {}", e),
                    })?;
            let d = LinearAlgebraAlgorithms::solve(client, q, &neg_grad_col)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: solve Q*d=-g - {}", e),
                })?
                .reshape(&[n])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: reshape d - {}", e),
                })?;

            // Step length limited by inactive inequality constraints
            let alpha = compute_max_step(client, &x, &d, a_ineq, b_ineq, &active_set)?;
            let step = client
                .mul_scalar(&d, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: scale d - {}", e),
                })?;
            x = client
                .add(&x, &step)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: x + step - {}", e),
                })?;

            // Activate any newly binding constraints
            if alpha < 1.0 {
                activate_binding(client, &x, a_ineq, b_ineq, &mut active_set, options.tol)?;
            }
        } else {
            // Solve equality-constrained QP subproblem
            let (working_a, _working_b) =
                build_working_set(client, a_eq, b_eq, a_ineq, b_ineq, &active_rows, n)?;

            // Solve KKT system for the direction
            let neg_grad =
                client
                    .mul_scalar(&grad, -1.0)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("active_set: negate grad - {}", e),
                    })?;

            // Build KKT: [Q A'] [d]      = [-grad]
            //             [A 0 ] [lambda]   [0    ]
            let size = n + m_working;
            let a_t = working_a
                .transpose(0, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: transpose A - {}", e),
                })?;

            let zeros_mm = client
                .fill(&[m_working, m_working], 0.0, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: zeros - {}", e),
                })?;

            let top = client
                .cat(&[q, &a_t], 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: cat top - {}", e),
                })?;
            let bottom = client.cat(&[&working_a, &zeros_mm], 1).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("active_set: cat bottom - {}", e),
                }
            })?;
            let kkt =
                client
                    .cat(&[&top, &bottom], 0)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("active_set: cat kkt - {}", e),
                    })?;

            let zeros_m = client.fill(&[m_working], 0.0, DType::F64).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("active_set: zeros rhs - {}", e),
                }
            })?;
            let rhs = client.cat(&[&neg_grad, &zeros_m], 0).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("active_set: cat rhs - {}", e),
                }
            })?;
            let rhs_col = rhs
                .reshape(&[size, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: reshape rhs - {}", e),
                })?;

            let sol = LinearAlgebraAlgorithms::solve(client, &kkt, &rhs_col).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("active_set: solve kkt - {}", e),
                }
            })?;
            let sol_flat = sol
                .reshape(&[size])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: reshape sol - {}", e),
                })?;

            let d = sol_flat
                .narrow(0, 0, n)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: extract d - {}", e),
                })?
                .contiguous();
            let lambdas = sol_flat
                .narrow(0, n, m_working)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: extract lambda - {}", e),
                })?
                .contiguous();

            let d_norm = tensor_norm(client, &d).map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: d norm - {}", e),
            })?;

            if d_norm < options.tol {
                // At stationary point of subproblem. Check multipliers.
                // Extract inequality multipliers (skip equality multipliers)
                let ineq_lambdas = if m_eq > 0 {
                    lambdas
                        .narrow(0, m_eq, lambdas.shape()[0] - m_eq)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("active_set: extract ineq lambdas - {}", e),
                        })?
                        .contiguous()
                } else {
                    lambdas.clone()
                };

                // Find minimum value (most negative multiplier)
                let min_lambda = client.min(&ineq_lambdas, &[0], false).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("active_set: min lambda - {}", e),
                    }
                })?;
                let min_val: f64 = min_lambda.to_vec()[0];

                if min_val < -1e-10 {
                    // Find index of most negative multiplier
                    let idx_tensor = client.argmin(&ineq_lambdas, 0, false).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("active_set: argmin lambda - {}", e),
                        }
                    })?;
                    let idx_vals: Vec<f64> = idx_tensor.to_vec();
                    let idx_val = idx_vals[0] as i64;
                    let row_i = active_rows[idx_val as usize];
                    active_set[row_i] = false;
                } else {
                    // All multipliers non-negative -> optimal
                    let fun = compute_qp_objective(client, q, c_vec, &x, n)?;
                    return Ok(QpResult {
                        x,
                        fun,
                        iterations: iter + 1,
                        converged: true,
                        dual_eq: if m_eq > 0 {
                            Some(
                                lambdas
                                    .narrow(0, 0, m_eq)
                                    .map_err(|e| OptimizeError::NumericalError {
                                        message: format!("active_set: extract dual_eq - {}", e),
                                    })?
                                    .contiguous(),
                            )
                        } else {
                            None
                        },
                        dual_ineq: None,
                    });
                }
            } else {
                // Take step in direction d
                let alpha = compute_max_step(client, &x, &d, a_ineq, b_ineq, &active_set)?;
                let step =
                    client
                        .mul_scalar(&d, alpha)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("active_set: scale d - {}", e),
                        })?;
                x = client
                    .add(&x, &step)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("active_set: x + step - {}", e),
                    })?;

                if alpha < 1.0 {
                    activate_binding(client, &x, a_ineq, b_ineq, &mut active_set, options.tol)?;
                }
            }
        }
    }

    let fun = compute_qp_objective(client, q, c_vec, &x, n)?;
    Ok(QpResult {
        x,
        fun,
        iterations: options.max_iter,
        converged: false,
        dual_eq: None,
        dual_ineq: None,
    })
}

fn compute_qp_objective<R, C>(
    client: &C,
    q: &Tensor<R>,
    c: &Tensor<R>,
    x: &Tensor<R>,
    n: usize,
) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let x_col = x
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_obj: reshape x - {}", e),
        })?;
    let qx = client
        .matmul(q, &x_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_obj: Q*x - {}", e),
        })?
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("qp_obj: reshape Qx - {}", e),
        })?;
    let xtqx = tensor_dot(client, x, &qx).map_err(|e| OptimizeError::NumericalError {
        message: format!("qp_obj: x'Qx - {}", e),
    })?;
    let ctx = tensor_dot(client, c, x).map_err(|e| OptimizeError::NumericalError {
        message: format!("qp_obj: c'x - {}", e),
    })?;
    Ok(0.5 * xtqx + ctx)
}

fn compute_max_step<R, C>(
    client: &C,
    x: &Tensor<R>,
    d: &Tensor<R>,
    a_ineq: Option<&Tensor<R>>,
    b_ineq: Option<&Tensor<R>>,
    active_set: &[bool],
) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let Some(a) = a_ineq else { return Ok(1.0) };
    let b = b_ineq.ok_or(OptimizeError::InvalidInput {
        context: "max_step: b_ineq must be provided when a_ineq is present".to_string(),
    })?;

    let n = d.shape()[0];
    let m = a.shape()[0];

    let d_col = d
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: reshape d - {}", e),
        })?;
    let ad = client
        .matmul(a, &d_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: A*d - {}", e),
        })?
        .reshape(&[m])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: reshape Ad - {}", e),
        })?;

    let x_col = x
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: reshape x - {}", e),
        })?;
    let ax = client
        .matmul(a, &x_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: A*x - {}", e),
        })?
        .reshape(&[m])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("max_step: reshape Ax - {}", e),
        })?;

    let ad_vals: Vec<f64> = ad.to_vec();
    let ax_vals: Vec<f64> = ax.to_vec();
    let b_vals: Vec<f64> = b.to_vec();

    let mut alpha = 1.0;
    for i in 0..m {
        if active_set[i] {
            continue;
        }
        // Constraint: a_i'*x >= b_i => a_i'*(x+alpha*d) >= b_i
        // alpha <= (b_i - a_i'*x) / a_i'*d when a_i'*d < 0
        if ad_vals[i] < -1e-15 {
            let limit = (b_vals[i] - ax_vals[i]) / ad_vals[i];
            if limit >= 0.0 && limit < alpha {
                alpha = limit;
            }
        }
    }
    Ok(alpha)
}

fn activate_binding<R, C>(
    client: &C,
    x: &Tensor<R>,
    a_ineq: Option<&Tensor<R>>,
    b_ineq: Option<&Tensor<R>>,
    active_set: &mut [bool],
    tol: f64,
) -> OptimizeResult<()>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let Some(a) = a_ineq else { return Ok(()) };
    let b = b_ineq.ok_or(OptimizeError::InvalidInput {
        context: "activate_binding: b_ineq must be provided when a_ineq is present".to_string(),
    })?;

    let n = x.shape()[0];
    let m = a.shape()[0];

    let x_col = x
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("activate: reshape x - {}", e),
        })?;
    let ax = client
        .matmul(a, &x_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("activate: A*x - {}", e),
        })?
        .reshape(&[m])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("activate: reshape Ax - {}", e),
        })?;

    let residual = client
        .sub(&ax, b)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("activate: Ax - b - {}", e),
        })?;
    let vals: Vec<f64> = residual.to_vec();

    for (i, &v) in vals.iter().enumerate() {
        if !active_set[i] && v.abs() < tol {
            active_set[i] = true;
        }
    }
    Ok(())
}

fn build_working_set<R, C>(
    client: &C,
    a_eq: Option<&Tensor<R>>,
    b_eq: Option<&Tensor<R>>,
    a_ineq: Option<&Tensor<R>>,
    b_ineq: Option<&Tensor<R>>,
    active_rows: &[usize],
    _n: usize,
) -> OptimizeResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut a_parts: Vec<Tensor<R>> = Vec::new();
    let mut b_parts: Vec<Tensor<R>> = Vec::new();

    if let (Some(ae), Some(be)) = (a_eq, b_eq) {
        a_parts.push(ae.clone());
        b_parts.push(be.clone());
    }

    if let (Some(ai), Some(bi)) = (a_ineq, b_ineq) {
        for &row in active_rows {
            let a_row = ai
                .narrow(0, row, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("working_set: narrow A {} - {}", row, e),
                })?
                .contiguous();
            let b_val = bi
                .narrow(0, row, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("working_set: narrow b {} - {}", row, e),
                })?
                .contiguous();
            a_parts.push(a_row);
            b_parts.push(b_val);
        }
    }

    let a_refs: Vec<&Tensor<R>> = a_parts.iter().collect();
    let b_refs: Vec<&Tensor<R>> = b_parts.iter().collect();

    let combined_a = client
        .cat(&a_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("working_set: cat A - {}", e),
        })?;
    let combined_b = client
        .cat(&b_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("working_set: cat b - {}", e),
        })?;

    Ok((combined_a, combined_b))
}
