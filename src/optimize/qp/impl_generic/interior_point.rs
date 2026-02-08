//! Interior point method for quadratic programming.
//!
//! Predictor-corrector Mehrotra scheme.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::impl_generic::utils::tensor_dot;
use crate::optimize::qp::traits::{QpOptions, QpResult};

/// Interior point QP solver using Mehrotra predictor-corrector.
///
/// Transforms inequality constraints A_ineq*x >= b_ineq into standard form
/// with slack variables: A_ineq*x - s = b_ineq, s >= 0.
/// Then solves the barrier KKT system.
#[allow(clippy::too_many_arguments)]
pub fn interior_point_qp_impl<R, C>(
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

    // If no inequality constraints, solve directly via KKT
    if m_ineq == 0 {
        return solve_equality_only_qp(client, q, c_vec, a_eq, b_eq, n, m_eq, options);
    }

    let ai = a_ineq.ok_or(OptimizeError::InvalidInput {
        context: "interior_point: a_ineq must be present when m_ineq > 0".to_string(),
    })?;
    let bi = b_ineq.ok_or(OptimizeError::InvalidInput {
        context: "interior_point: b_ineq must be present when m_ineq > 0".to_string(),
    })?;

    // Initialize primal x, slack s, dual z
    let mut x = client
        .fill(&[n], 0.0, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("ip_qp: initial x - {}", e),
        })?;

    // Initialize slacks (s > 0)
    let mut s =
        client
            .fill(&[m_ineq], 1.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: initial s - {}", e),
            })?;

    // Initialize dual variables (z > 0)
    let mut z =
        client
            .fill(&[m_ineq], 1.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: initial z - {}", e),
            })?;

    for iter in 0..options.max_iter {
        // Compute complementarity gap: mu = s'z / m_ineq
        let sz = tensor_dot(client, &s, &z).map_err(|e| OptimizeError::NumericalError {
            message: format!("ip_qp: s'z - {}", e),
        })?;
        let mu = sz / m_ineq as f64;

        if mu < options.tol {
            let fun = compute_objective(client, q, c_vec, &x, n)?;
            return Ok(QpResult {
                x,
                fun,
                iterations: iter + 1,
                converged: true,
                dual_eq: None,
                dual_ineq: Some(z),
            });
        }

        // Compute residuals
        // r_dual = Q*x + c - A_ineq'*z  (- A_eq'*y if eq constraints)
        let x_col = x
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape x - {}", e),
            })?;
        let qx = client
            .matmul(q, &x_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: Q*x - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape Qx - {}", e),
            })?;

        let ai_t = ai
            .transpose(0, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: transpose A - {}", e),
            })?;
        let z_col = z
            .reshape(&[m_ineq, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape z - {}", e),
            })?;
        let atz = client
            .matmul(&ai_t, &z_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: A'z - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape A'z - {}", e),
            })?;

        let r_dual = client
            .add(&qx, c_vec)
            .and_then(|r| client.sub(&r, &atz))
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: r_dual - {}", e),
            })?;

        // r_primal = A*x - s - b
        let ax = client
            .matmul(ai, &x_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: A*x - {}", e),
            })?
            .reshape(&[m_ineq])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape Ax - {}", e),
            })?;
        let r_primal = client
            .sub(&ax, &s)
            .and_then(|r| client.sub(&r, bi))
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: r_primal - {}", e),
            })?;

        // Centering parameter
        let sigma = 0.3;
        let sigma_mu = sigma * mu;

        // r_comp = S*Z*e - sigma*mu*e  (where S=diag(s), Z=diag(z))
        let sz_elem = client
            .mul(&s, &z)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: s.*z - {}", e),
            })?;
        let sigma_mu_vec = client.fill(&[m_ineq], sigma_mu, DType::F64).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("ip_qp: sigma_mu vec - {}", e),
            }
        })?;
        let r_comp =
            client
                .sub(&sz_elem, &sigma_mu_vec)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: r_comp - {}", e),
                })?;

        // Solve Newton system by eliminating ds and dz:
        // dz = (r_comp - z.*ds) / s
        // ds = A*dx - r_primal
        // (Q + A'*(Z/S)*A) dx = -(r_dual + A'*(r_comp/s - z.*r_primal/s))
        let z_over_s = client
            .div(&z, &s)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: z/s - {}", e),
            })?;

        // Build reduced system
        let z_over_s_diag = LinearAlgebraAlgorithms::diagflat(client, &z_over_s).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("ip_qp: diagflat z/s - {}", e),
            }
        })?;
        let ai_t_zs =
            client
                .matmul(&ai_t, &z_over_s_diag)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: A'*(Z/S) - {}", e),
                })?;
        let ai_t_zs_ai =
            client
                .matmul(&ai_t_zs, ai)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: A'*(Z/S)*A - {}", e),
                })?;
        let lhs = client
            .add(q, &ai_t_zs_ai)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: Q + A'*(Z/S)*A - {}", e),
            })?;

        // Build RHS
        let rc_over_s = client
            .div(&r_comp, &s)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: r_comp/s - {}", e),
            })?;
        let z_rp_over_s =
            client
                .mul(&z_over_s, &r_primal)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: z*r_primal/s - {}", e),
                })?;
        let rhs_inner =
            client
                .sub(&rc_over_s, &z_rp_over_s)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: rc/s - z*rp/s - {}", e),
                })?;
        let rhs_inner_col =
            rhs_inner
                .reshape(&[m_ineq, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: reshape rhs_inner - {}", e),
                })?;
        let at_rhs = client
            .matmul(&ai_t, &rhs_inner_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: A'*rhs_inner - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape A'rhs - {}", e),
            })?;

        let neg_rhs = client
            .add(&r_dual, &at_rhs)
            .and_then(|r| client.mul_scalar(&r, -1.0))
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: neg rhs - {}", e),
            })?;
        let neg_rhs_col = neg_rhs
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape neg_rhs - {}", e),
            })?;

        // Solve for dx
        let dx = LinearAlgebraAlgorithms::solve(client, &lhs, &neg_rhs_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: solve dx - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape dx - {}", e),
            })?;

        // Recover ds = A*dx - r_primal
        let dx_col = dx
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape dx - {}", e),
            })?;
        let a_dx = client
            .matmul(ai, &dx_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: A*dx - {}", e),
            })?
            .reshape(&[m_ineq])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: reshape A*dx - {}", e),
            })?;
        let ds = client
            .sub(&a_dx, &r_primal)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: ds - {}", e),
            })?;

        // Recover dz = (r_comp - z.*ds) / s
        let z_ds = client
            .mul(&z, &ds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: z.*ds - {}", e),
            })?;
        let dz_num = client
            .sub(&r_comp, &z_ds)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: r_comp - z.*ds - {}", e),
            })?;
        let dz = client
            .div(&dz_num, &s)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: dz - {}", e),
            })?;

        // Step length (ensure s > 0 and z > 0)
        let alpha = compute_step_length(&s, &z, &ds, &dz)?;

        // Update
        let dx_scaled =
            client
                .mul_scalar(&dx, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: scale dx - {}", e),
                })?;
        let ds_scaled =
            client
                .mul_scalar(&ds, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: scale ds - {}", e),
                })?;
        let dz_scaled =
            client
                .mul_scalar(&dz, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("ip_qp: scale dz - {}", e),
                })?;

        x = client
            .add(&x, &dx_scaled)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: x update - {}", e),
            })?;
        s = client
            .add(&s, &ds_scaled)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: s update - {}", e),
            })?;
        z = client
            .add(&z, &dz_scaled)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("ip_qp: z update - {}", e),
            })?;
    }

    let fun = compute_objective(client, q, c_vec, &x, n)?;
    Ok(QpResult {
        x,
        fun,
        iterations: options.max_iter,
        converged: false,
        dual_eq: None,
        dual_ineq: Some(z),
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_equality_only_qp<R, C>(
    client: &C,
    q: &Tensor<R>,
    c_vec: &Tensor<R>,
    a_eq: Option<&Tensor<R>>,
    b_eq: Option<&Tensor<R>>,
    n: usize,
    m_eq: usize,
    _options: &QpOptions,
) -> OptimizeResult<QpResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    if m_eq == 0 {
        // Unconstrained: Q*x = -c
        let neg_c = client
            .mul_scalar(c_vec, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("eq_qp: negate c - {}", e),
            })?;
        let neg_c_col = neg_c
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("eq_qp: reshape c - {}", e),
            })?;
        let x = LinearAlgebraAlgorithms::solve(client, q, &neg_c_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("eq_qp: solve - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("eq_qp: reshape x - {}", e),
            })?;
        let fun = compute_objective(client, q, c_vec, &x, n)?;
        return Ok(QpResult {
            x,
            fun,
            iterations: 1,
            converged: true,
            dual_eq: None,
            dual_ineq: None,
        });
    }

    // KKT system
    let ae = a_eq.ok_or(OptimizeError::InvalidInput {
        context: "solve_equality_only_qp: a_eq must be present when m_eq > 0".to_string(),
    })?;
    let be = b_eq.ok_or(OptimizeError::InvalidInput {
        context: "solve_equality_only_qp: b_eq must be present when m_eq > 0".to_string(),
    })?;
    let size = n + m_eq;

    let a_t = ae
        .transpose(0, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: transpose A - {}", e),
        })?;
    let zeros =
        client
            .fill(&[m_eq, m_eq], 0.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("eq_qp: zeros - {}", e),
            })?;

    let top = client
        .cat(&[q, &a_t], 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: cat top - {}", e),
        })?;
    let bottom = client
        .cat(&[ae, &zeros], 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: cat bottom - {}", e),
        })?;
    let kkt = client
        .cat(&[&top, &bottom], 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: cat kkt - {}", e),
        })?;

    let neg_c = client
        .mul_scalar(c_vec, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: negate c - {}", e),
        })?;
    let rhs = client
        .cat(&[&neg_c, be], 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: cat rhs - {}", e),
        })?;
    let rhs_col = rhs
        .reshape(&[size, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: reshape rhs - {}", e),
        })?;

    let sol = LinearAlgebraAlgorithms::solve(client, &kkt, &rhs_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: solve kkt - {}", e),
        })?
        .reshape(&[size])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: reshape sol - {}", e),
        })?;

    let x = sol
        .narrow(0, 0, n)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: extract x - {}", e),
        })?
        .contiguous();
    let dual = sol
        .narrow(0, n, m_eq)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("eq_qp: extract dual - {}", e),
        })?
        .contiguous();

    let fun = compute_objective(client, q, c_vec, &x, n)?;
    Ok(QpResult {
        x,
        fun,
        iterations: 1,
        converged: true,
        dual_eq: Some(dual),
        dual_ineq: None,
    })
}

fn compute_objective<R, C>(
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
            message: format!("qp_obj: reshape - {}", e),
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

fn compute_step_length<R>(
    s: &Tensor<R>,
    z: &Tensor<R>,
    ds: &Tensor<R>,
    dz: &Tensor<R>,
) -> OptimizeResult<f64>
where
    R: Runtime,
{
    // This function computes the maximum step length to maintain s > 0 and z > 0.
    // We need: alpha <= -tau * s[i] / ds[i] when ds[i] < 0
    //          alpha <= -tau * z[i] / dz[i] when dz[i] < 0
    //
    // Since we need element-wise conditionals followed by reduction, and the problem
    // size is small (number of inequality constraints), we extract to Vec for this
    // scalar reduction operation. This is a control-flow computation (finding step bounds),
    // not a main algorithm computation, so the transfer overhead is acceptable.

    let s_vals: Vec<f64> = s.to_vec();
    let z_vals: Vec<f64> = z.to_vec();
    let ds_vals: Vec<f64> = ds.to_vec();
    let dz_vals: Vec<f64> = dz.to_vec();

    let tau = 0.995;
    let mut alpha: f64 = 1.0;

    for i in 0..s_vals.len() {
        if ds_vals[i] < 0.0 {
            alpha = alpha.min(-tau * s_vals[i] / ds_vals[i]);
        }
        if dz_vals[i] < 0.0 {
            alpha = alpha.min(-tau * z_vals[i] / dz_vals[i]);
        }
    }

    Ok(alpha.max(1e-15f64))
}
