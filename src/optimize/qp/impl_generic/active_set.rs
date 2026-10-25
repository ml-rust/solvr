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
    R: Runtime<DType = DType>,
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
                // Every inequality is inactive, so every multiplier is zero.
                let dual_ineq = scatter_ineq_duals(client, None, &[], m_ineq)?;
                return Ok(QpResult {
                    x,
                    fun,
                    iterations: iter + 1,
                    converged: true,
                    dual_eq: None,
                    dual_ineq,
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

            // Dependent working constraints make the KKT matrix singular.
            // The system stays consistent, so the min-norm solution is exact.
            let sol = LinearAlgebraAlgorithms::lstsq(client, &kkt, &rhs_col).map_err(|e| {
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
                .contiguous()
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: contiguous d - {}", e),
                })?;
            let lambdas = sol_flat
                .narrow(0, n, m_working)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: extract lambda - {}", e),
                })?
                .contiguous()
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: contiguous lambda - {}", e),
                })?;

            let d_norm = tensor_norm(client, &d).map_err(|e| OptimizeError::NumericalError {
                message: format!("active_set: d norm - {}", e),
            })?;

            if d_norm < options.tol {
                // At stationary point of subproblem. Check multipliers.
                // With no inequality in the working set, nothing can leave it.
                let ineq_mu = if m_working > m_eq {
                    // Extract inequality multipliers (skip equality multipliers)
                    let ineq_lambdas = lambdas
                        .narrow(0, m_eq, m_working - m_eq)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("active_set: extract ineq lambdas - {}", e),
                        })?
                        .contiguous()
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("active_set: contiguous ineq lambdas - {}", e),
                        })?;

                    // The KKT rows are Q d + A' lambda = -g, so the multiplier of
                    // a'x >= b is mu = -lambda. Optimal when every mu is non-negative.
                    Some(client.mul_scalar(&ineq_lambdas, -1.0).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("active_set: negate lambda - {}", e),
                        }
                    })?)
                } else {
                    None
                };

                let dropped_row = match &ineq_mu {
                    Some(mu) => {
                        let min_mu = client.min(mu, &[0], false).map_err(|e| {
                            OptimizeError::NumericalError {
                                message: format!("active_set: min lambda - {}", e),
                            }
                        })?;
                        let min_val: f64 = min_mu.to_vec()[0];

                        if min_val < -1e-10 {
                            // Find index of most negative multiplier
                            let idx_tensor = client.argmin(mu, 0, false).map_err(|e| {
                                OptimizeError::NumericalError {
                                    message: format!("active_set: argmin lambda - {}", e),
                                }
                            })?;
                            let idx_vals: Vec<i64> = idx_tensor.to_vec();
                            Some(active_rows[idx_vals[0] as usize])
                        } else {
                            None
                        }
                    }
                    None => None,
                };

                if let Some(row_i) = dropped_row {
                    active_set[row_i] = false;
                } else {
                    // All multipliers non-negative -> optimal
                    let fun = compute_qp_objective(client, q, c_vec, &x, n)?;
                    let dual_ineq =
                        scatter_ineq_duals(client, ineq_mu.as_ref(), &active_rows, m_ineq)?;
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
                                    .contiguous()
                                    .map_err(|e| OptimizeError::NumericalError {
                                        message: format!("active_set: contiguous dual_eq - {}", e),
                                    })?,
                            )
                        } else {
                            None
                        },
                        dual_ineq,
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
    R: Runtime<DType = DType>,
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

/// Inequality multipliers `[m_ineq]` from the multipliers of the active rows.
///
/// `mu[j]` belongs to row `active_rows[j]`. Every inactive row gets zero.
/// Returns `None` when the problem has no inequality constraints.
fn scatter_ineq_duals<R, C>(
    client: &C,
    mu: Option<&Tensor<R>>,
    active_rows: &[usize],
    m_ineq: usize,
) -> OptimizeResult<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    if m_ineq == 0 {
        return Ok(None);
    }
    let Some(mu) = mu else {
        let zeros =
            client
                .fill(&[m_ineq], 0.0, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("active_set: zero dual_ineq - {}", e),
                })?;
        return Ok(Some(zeros));
    };

    // Selection matrix P [m_ineq, k] with P[active_rows[j], j] = 1, so P * mu
    // places each active multiplier at its row. The row indices live on the host.
    let k = active_rows.len();
    let mut selection = vec![0.0f64; m_ineq * k];
    for (j, &row) in active_rows.iter().enumerate() {
        selection[row * k + j] = 1.0;
    }
    let p = Tensor::<R>::from_slice(&selection, &[m_ineq, k], client.device()).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("active_set: dual_ineq selection - {}", e),
        }
    })?;
    let mu_col = mu
        .reshape(&[k, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("active_set: reshape mu - {}", e),
        })?;
    let full = client
        .matmul(&p, &mu_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("active_set: scatter dual_ineq - {}", e),
        })?
        .reshape(&[m_ineq])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("active_set: reshape dual_ineq - {}", e),
        })?;
    Ok(Some(full))
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
    R: Runtime<DType = DType>,
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
    R: Runtime<DType = DType>,
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
    R: Runtime<DType = DType>,
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
                .contiguous()
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("working_set: contiguous A {} - {}", row, e),
                })?;
            let b_val = bi
                .narrow(0, row, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("working_set: narrow b {} - {}", row, e),
                })?
                .contiguous()
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("working_set: contiguous b {} - {}", row, e),
                })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    fn t(data: &[f64], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    fn eye3() -> Vec<f64> {
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn test_active_set_fewer_equalities_than_variables() {
        // min 0.5|x|^2 s.t. x1 + x2 + x3 = 3. One row, three unknowns.
        // Solution x = (1, 1, 1), objective 1.5, multiplier -1.
        let (device, client) = setup();
        let q = t(&eye3(), &[3, 3], &device);
        let c = t(&[0.0; 3], &[3], &device);
        let a = t(&[1.0, 1.0, 1.0], &[1, 3], &device);
        let b = t(&[3.0], &[1], &device);
        let opts = QpOptions::default();
        let r = active_set_qp_impl(&client, &q, &c, Some(&a), Some(&b), None, None, &opts).unwrap();
        assert!(r.converged);
        let x: Vec<f64> = r.x.to_vec();
        for xi in &x {
            assert!((xi - 1.0).abs() < 1e-8, "x = {:?}", x);
        }
        assert!((r.fun - 1.5).abs() < 1e-8, "fun = {}", r.fun);
        let lam: Vec<f64> = r.dual_eq.unwrap().to_vec();
        // Stationarity: Qx + c + A' lambda = 0.
        assert!((x[0] + lam[0]).abs() < 1e-8, "lambda = {:?}", lam);
    }

    #[test]
    fn test_active_set_dependent_equalities() {
        // Rows (1,1,1) and (2,2,2) are dependent and consistent (b = 3, 6).
        // Solution x = (1, 1, 1). Min-norm multipliers satisfy l1 + 2 l2 = -1.
        let (device, client) = setup();
        let q = t(&eye3(), &[3, 3], &device);
        let c = t(&[0.0; 3], &[3], &device);
        let a = t(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3], &device);
        let b = t(&[3.0, 6.0], &[2], &device);
        let opts = QpOptions::default();
        let r = active_set_qp_impl(&client, &q, &c, Some(&a), Some(&b), None, None, &opts).unwrap();
        assert!(r.converged);
        let x: Vec<f64> = r.x.to_vec();
        for xi in &x {
            assert!(xi.is_finite() && (xi - 1.0).abs() < 1e-8, "x = {:?}", x);
        }
        let lam: Vec<f64> = r.dual_eq.unwrap().to_vec();
        assert!(lam.iter().all(|l| l.is_finite()));
        assert!(
            (lam[0] + 2.0 * lam[1] + 1.0).abs() < 1e-8,
            "lambda = {:?}",
            lam
        );
        assert!((x.iter().sum::<f64>() - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_active_set_one_active_inequality_of_three_variables() {
        // min 0.5|x - (2,2,2)|^2 s.t. -(x1 + x2 + x3) >= -3.
        // Projection onto the half-space: x = (1, 1, 1), objective -4.5.
        // The origin is feasible. One active row, three unknowns.
        let (device, client) = setup();
        let q = t(&eye3(), &[3, 3], &device);
        let c = t(&[-2.0; 3], &[3], &device);
        let a = t(&[-1.0, -1.0, -1.0], &[1, 3], &device);
        let b = t(&[-3.0], &[1], &device);
        let opts = QpOptions {
            method: crate::optimize::qp::traits::QpMethod::ActiveSet,
            ..QpOptions::default()
        };
        let r = active_set_qp_impl(&client, &q, &c, None, None, Some(&a), Some(&b), &opts).unwrap();
        assert!(r.converged);
        let x: Vec<f64> = r.x.to_vec();
        for xi in &x {
            assert!((xi - 1.0).abs() < 1e-8, "x = {:?}", x);
        }
        assert!((r.fun + 4.5).abs() < 1e-8, "fun = {}", r.fun);
        let z: Vec<f64> = r.dual_ineq.unwrap().to_vec();
        // Stationarity: x + c - a' z = 0 gives (1 - 2) + z = 0, so z = 1.
        assert!((z[0] - 1.0).abs() < 1e-8, "z = {:?}", z);
    }

    #[test]
    fn test_active_set_drops_only_the_negative_second_multiplier() {
        // min 0.5|x|^2 + 2 x1 - x2  s.t.  row 0: x1 >= 0,  row 1: x1 + x2 >= 0.
        //
        // The origin lies on both rows, and the unconstrained step d = -c = (-2, 1)
        // leaves both, so both enter the working set at the origin. There
        // c = 3 a0 - 1 a1, so the multipliers are z = (3, -1). Only row 1 has a
        // negative multiplier and must leave. Dropping row 0 instead cycles at the
        // origin and never converges.
        //
        // Optimum with row 0 active: x = (0, 1), objective -0.5.
        // KKT: x + c - A' z = (2, 0) - z0 (1, 0) - z1 (1, 1) = 0 gives z = (2, 0).
        // Primal: x1 = 0 >= 0, x1 + x2 = 1 >= 0. Complementarity: z1 * 1 = 0.
        let (device, client) = setup();
        let q = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let c = t(&[2.0, -1.0], &[2], &device);
        let a = t(&[1.0, 0.0, 1.0, 1.0], &[2, 2], &device);
        let b = t(&[0.0, 0.0], &[2], &device);
        let opts = QpOptions {
            method: crate::optimize::qp::traits::QpMethod::ActiveSet,
            ..QpOptions::default()
        };
        let r = active_set_qp_impl(&client, &q, &c, None, None, Some(&a), Some(&b), &opts).unwrap();
        assert!(r.converged, "iterations = {}", r.iterations);

        let x: Vec<f64> = r.x.to_vec();
        assert!(
            x[0].abs() < 1e-8 && (x[1] - 1.0).abs() < 1e-8,
            "x = {:?}",
            x
        );
        assert!((r.fun + 0.5).abs() < 1e-8, "fun = {}", r.fun);

        let z: Vec<f64> = r.dual_ineq.unwrap().to_vec();
        assert_eq!(z.len(), 2);
        assert!(
            (z[0] - 2.0).abs() < 1e-8 && z[1].abs() < 1e-8,
            "z = {:?}",
            z
        );

        // Stationarity, dual feasibility, primal feasibility, complementarity.
        let stat = [x[0] + 2.0 - z[0] - z[1], x[1] - 1.0 - z[1]];
        assert!(
            stat.iter().all(|s| s.abs() < 1e-8),
            "stationarity = {:?}",
            stat
        );
        assert!(z.iter().all(|&zi| zi >= -1e-10), "z = {:?}", z);
        let slack = [x[0], x[0] + x[1]];
        assert!(slack.iter().all(|&s| s >= -1e-10), "slack = {:?}", slack);
        assert!((z[0] * slack[0]).abs() < 1e-8 && (z[1] * slack[1]).abs() < 1e-8);
    }
}
