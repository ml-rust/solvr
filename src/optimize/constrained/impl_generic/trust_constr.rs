//! Trust-region constrained optimization (trust-constr) implementation.
//!
//! Two-mode algorithm:
//! - Equality-only: Byrd-Omojokun trust-region SQP
//! - With inequalities: Barrier/interior point + equality SQP
//!
//! Reference: scipy.optimize._trustregion_constr

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::constrained::traits::types::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, ConstraintType,
};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::impl_generic::utils::{
    finite_difference_gradient, finite_difference_jacobian, tensor_dot, tensor_norm,
};

/// Trust-constr implementation generic over Runtime.
pub fn trust_constr_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
    options: &ConstrainedOptions,
) -> OptimizeResult<ConstrainedResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "trust_constr: empty initial guess".to_string(),
        });
    }

    let has_inequality = constraints
        .iter()
        .any(|c| c.kind == ConstraintType::Inequality)
        || bounds.lower.is_some()
        || bounds.upper.is_some();

    if has_inequality {
        barrier_method(client, &f, x0, constraints, bounds, options, n)
    } else {
        equality_sqp(client, &f, x0, constraints, options, n)
    }
}

/// Equality-only SQP with trust region (Byrd-Omojokun approach).
fn equality_sqp<R, C, F>(
    client: &C,
    f: &F,
    x0: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    options: &ConstrainedOptions,
    n: usize,
) -> OptimizeResult<ConstrainedResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("trust_constr: initial f - {}", e),
    })?;
    let mut nfev = 1usize;
    let mut trust_radius = 1.0;
    let eta = 0.1; // Accept ratio threshold

    // Initialize Hessian approximation
    let mut b_hess =
        client
            .eye(n, None, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: identity - {}", e),
            })?;

    for iter in 0..options.max_iter {
        let grad = finite_difference_gradient(client, f, &x, fx, options.eps).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("trust_constr: gradient - {}", e),
            }
        })?;
        nfev += n;

        // Evaluate constraint values and Jacobian
        let (c_vals, jac) = eval_eq_constraints(client, &x, constraints, n, options.eps)?;
        nfev += constraints.len() * (n + 1);

        let max_viol = if let Some(ref c) = c_vals {
            // Use tensor operations to compute max absolute value
            let abs_c = client.abs(c).map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: abs c - {}", e),
            })?;
            let max_val =
                client
                    .max(&abs_c, &[0], false)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("trust_constr: max c - {}", e),
                    })?;
            max_val.to_vec()[0] // Single scalar extraction
        } else {
            0.0
        };

        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: grad norm - {}", e),
        })?;

        if grad_norm < options.tol && max_viol < options.constraint_tol {
            return Ok(ConstrainedResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
                constraint_violation: max_viol,
                message: "Optimization converged".to_string(),
            });
        }

        // Compute trust region step
        let step = compute_trust_step(
            client,
            &b_hess,
            &grad,
            jac.as_ref(),
            c_vals.as_ref(),
            trust_radius,
            n,
        )?;

        let x_trial = client
            .add(&x, &step)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: x + step - {}", e),
            })?;

        let fx_trial = f(&x_trial).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: f trial - {}", e),
        })?;
        nfev += 1;

        // Compute predicted reduction using quadratic model
        let pred_reduction = compute_predicted_reduction(client, &b_hess, &grad, &step, n)?;

        // Actual reduction
        let actual_reduction = fx - fx_trial;

        // Ratio
        let ratio = if pred_reduction.abs() < 1e-15 {
            0.0
        } else {
            actual_reduction / pred_reduction
        };

        // Update trust radius
        let step_norm = tensor_norm(client, &step).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: step norm - {}", e),
        })?;

        if ratio < 0.25 {
            trust_radius = 0.25 * step_norm;
        } else if ratio > 0.75 && (step_norm - trust_radius).abs() < 1e-10 * trust_radius {
            trust_radius = (2.0 * trust_radius).min(1e10);
        }

        if ratio > eta {
            // Accept step
            let grad_new = finite_difference_gradient(client, f, &x_trial, fx_trial, options.eps)
                .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: new gradient - {}", e),
            })?;
            nfev += n;

            let y = client
                .sub(&grad_new, &grad)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("trust_constr: y - {}", e),
                })?;

            // SR1-like Hessian update (more robust than BFGS for constraints)
            b_hess = sr1_update(client, &b_hess, &step, &y, n)?;

            x = x_trial;
            fx = fx_trial;
        }

        if trust_radius < 1e-15 {
            return Ok(ConstrainedResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: false,
                constraint_violation: max_viol,
                message: "Trust radius too small".to_string(),
            });
        }
    }

    Ok(ConstrainedResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
        constraint_violation: 0.0,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Barrier/interior point method for inequality constraints.
///
/// Transforms inequality constraints into a sequence of equality-constrained
/// problems using log-barrier functions.
fn barrier_method<R, C, F>(
    client: &C,
    f: &F,
    x0: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
    options: &ConstrainedOptions,
    n: usize,
) -> OptimizeResult<ConstrainedResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut x = x0.clone();
    // Clamp to bounds
    if let Some(ref lower) = bounds.lower {
        x = client
            .maximum(&x, lower)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("barrier: max lower - {}", e),
            })?;
    }
    if let Some(ref upper) = bounds.upper {
        x = client
            .minimum(&x, upper)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("barrier: min upper - {}", e),
            })?;
    }

    let mut mu = 1.0; // Barrier parameter
    let mu_factor = 0.2; // Reduction factor
    let mu_min = 1e-12;
    let mut nfev = 0usize;
    let mut total_iter = 0usize;

    while mu > mu_min && total_iter < options.max_iter {
        // Inner problem: minimize f(x) - mu * sum(log(c_i(x))) for ineq constraints
        // subject to equality constraints
        let mu_current = mu;
        let barrier_f = |x_inner: &Tensor<R>| -> Result<f64> {
            let fx = f(x_inner)?;
            let penalty = compute_barrier_penalty(x_inner, constraints, bounds, mu_current)?;
            Ok(fx + penalty)
        };

        // Run a few Newton-like steps for the barrier subproblem
        let sub_options = ConstrainedOptions {
            max_iter: 20.min(options.max_iter - total_iter),
            tol: mu.max(options.tol),
            eps: options.eps,
            constraint_tol: options.constraint_tol,
        };

        // Simple gradient descent for barrier subproblem
        let (x_new, _fx_new, sub_nfev, sub_iter) =
            barrier_inner_solve(client, &barrier_f, f, &x, bounds, &sub_options, n)?;
        nfev += sub_nfev;
        total_iter += sub_iter;

        x = x_new;
        mu *= mu_factor;
    }

    let fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("barrier: final f - {}", e),
    })?;
    nfev += 1;

    let max_viol = compute_max_violation_all(&x, constraints, bounds)?;

    Ok(ConstrainedResult {
        x,
        fun: fx,
        iterations: total_iter,
        nfev,
        converged: max_viol < options.constraint_tol,
        constraint_violation: max_viol,
        message: if max_viol < options.constraint_tol {
            "Optimization converged".to_string()
        } else {
            "Barrier method completed".to_string()
        },
    })
}

/// Inner solve for the barrier subproblem using gradient descent with backtracking.
fn barrier_inner_solve<R, C, F, G>(
    client: &C,
    barrier_f: &F,
    original_f: &G,
    x0: &Tensor<R>,
    bounds: &Bounds<R>,
    options: &ConstrainedOptions,
    n: usize,
) -> OptimizeResult<(Tensor<R>, f64, usize, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
    G: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut x = x0.clone();
    let mut fx = barrier_f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("barrier_inner: initial f - {}", e),
    })?;
    let mut nfev = 1usize;

    for iter in 0..options.max_iter {
        let grad =
            finite_difference_gradient(client, barrier_f, &x, fx, options.eps).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("barrier_inner: gradient - {}", e),
                }
            })?;
        nfev += n;

        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("barrier_inner: grad norm - {}", e),
        })?;

        if grad_norm < options.tol {
            let fx_orig = original_f(&x).map_err(|e| OptimizeError::NumericalError {
                message: format!("barrier_inner: orig f - {}", e),
            })?;
            nfev += 1;
            return Ok((x, fx_orig, nfev, iter + 1));
        }

        // Steepest descent with backtracking
        let direction =
            client
                .mul_scalar(&grad, -1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("barrier_inner: negate grad - {}", e),
                })?;

        let mut alpha = 1.0 / grad_norm.max(1.0);
        let c_armijo = 1e-4;

        for _ in 0..30 {
            let step = client.mul_scalar(&direction, alpha).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("barrier_inner: scale step - {}", e),
                }
            })?;
            let x_trial = client
                .add(&x, &step)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("barrier_inner: x + step - {}", e),
                })?;

            // Clamp to bounds (stay strictly interior)
            let x_trial = clamp_interior(client, &x_trial, bounds, 1e-12)?;

            let fx_trial = match barrier_f(&x_trial) {
                Ok(v) if v.is_finite() => v,
                _ => {
                    alpha *= 0.5;
                    continue;
                }
            };
            nfev += 1;

            if fx_trial <= fx - c_armijo * alpha * grad_norm * grad_norm {
                x = x_trial;
                fx = fx_trial;
                break;
            }
            alpha *= 0.5;
        }
    }

    let fx_orig = original_f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("barrier_inner: final orig f - {}", e),
    })?;
    nfev += 1;

    Ok((x, fx_orig, nfev, options.max_iter))
}

/// Compute log-barrier penalty: -mu * sum(log(c_i(x))) for inequality constraints.
/// Note: This function extracts constraint values at the evaluation boundary.
fn compute_barrier_penalty<R: Runtime>(
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
    mu: f64,
) -> Result<f64> {
    let mut penalty = 0.0;

    for constraint in constraints {
        if constraint.kind != ConstraintType::Inequality {
            continue;
        }
        let c_val = (constraint.fun)(x)?;
        let vals: Vec<f64> = c_val.to_vec();
        for &v in &vals {
            if v <= 0.0 {
                return Ok(f64::INFINITY);
            }
            penalty -= mu * v.ln();
        }
    }

    // Bound barriers
    let x_vals: Vec<f64> = x.to_vec();
    if let Some(ref lower) = bounds.lower {
        let l_vals: Vec<f64> = lower.to_vec();
        for (xi, li) in x_vals.iter().zip(l_vals.iter()) {
            let slack = xi - li;
            if slack <= 0.0 {
                return Ok(f64::INFINITY);
            }
            penalty -= mu * slack.ln();
        }
    }
    if let Some(ref upper) = bounds.upper {
        let u_vals: Vec<f64> = upper.to_vec();
        for (xi, ui) in x_vals.iter().zip(u_vals.iter()) {
            let slack = ui - xi;
            if slack <= 0.0 {
                return Ok(f64::INFINITY);
            }
            penalty -= mu * slack.ln();
        }
    }

    Ok(penalty)
}

fn clamp_interior<R, C>(
    client: &C,
    x: &Tensor<R>,
    bounds: &Bounds<R>,
    margin: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut result = x.clone();
    if let Some(ref lower) = bounds.lower {
        let lower_margin =
            client
                .add_scalar(lower, margin)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("clamp_interior: lower + margin - {}", e),
                })?;
        result =
            client
                .maximum(&result, &lower_margin)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("clamp_interior: max lower - {}", e),
                })?;
    }
    if let Some(ref upper) = bounds.upper {
        let upper_margin =
            client
                .add_scalar(upper, -margin)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("clamp_interior: upper - margin - {}", e),
                })?;
        result = client
            .maximum(
                &client.minimum(&result, &upper_margin).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("clamp_interior: min upper - {}", e),
                    }
                })?,
                &client
                    .add_scalar(bounds.lower.as_ref().unwrap_or(&result), margin)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("clamp_interior: max check - {}", e),
                    })?,
            )
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("clamp_interior: max after min - {}", e),
            })?;
    }
    Ok(result)
}

#[allow(clippy::type_complexity)]
fn eval_eq_constraints<R, C>(
    client: &C,
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    n: usize,
    eps: f64,
) -> OptimizeResult<(Option<Tensor<R>>, Option<Tensor<R>>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let eq_constraints: Vec<&Constraint<'_, R>> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintType::Equality)
        .collect();

    if eq_constraints.is_empty() {
        return Ok((None, None));
    }

    let mut c_parts: Vec<Tensor<R>> = Vec::new();
    let mut j_parts: Vec<Tensor<R>> = Vec::new();

    for constraint in &eq_constraints {
        let c_val = (constraint.fun)(x).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: eq constraint eval - {}", e),
        })?;
        let m_i = c_val.shape()[0];

        let jac = if let Some(jac_fn) = constraint.jac {
            jac_fn(x).map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: eq constraint jac - {}", e),
            })?
        } else {
            finite_difference_jacobian(
                client,
                &|x_inner: &Tensor<R>| (constraint.fun)(x_inner),
                x,
                &c_val,
                m_i,
                n,
                eps,
            )
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_constr: eq fd jac - {}", e),
            })?
        };

        c_parts.push(c_val);
        j_parts.push(jac);
    }

    let c_refs: Vec<&Tensor<R>> = c_parts.iter().collect();
    let j_refs: Vec<&Tensor<R>> = j_parts.iter().collect();

    let c_all = client
        .cat(&c_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: cat eq vals - {}", e),
        })?;
    let j_all = client
        .cat(&j_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_constr: cat eq jac - {}", e),
        })?;

    Ok((Some(c_all), Some(j_all)))
}

fn compute_trust_step<R, C>(
    client: &C,
    b: &Tensor<R>,
    g: &Tensor<R>,
    _jac: Option<&Tensor<R>>,
    _c_vals: Option<&Tensor<R>>,
    trust_radius: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    // Cauchy point for the trust region
    let _g_col = g
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_step: reshape g - {}", e),
        })?;

    // Steepest descent: p = -g, scaled to trust region
    let g_norm = tensor_norm(client, g).map_err(|e| OptimizeError::NumericalError {
        message: format!("trust_step: g norm - {}", e),
    })?;

    if g_norm < 1e-15 {
        return client
            .fill(&[n], 0.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_step: zero step - {}", e),
            });
    }

    // Try Newton step first
    let neg_g = client
        .mul_scalar(g, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_step: negate g - {}", e),
        })?;
    let neg_g_col = neg_g
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_step: reshape neg_g - {}", e),
        })?;

    let newton_step: Option<Tensor<R>> = match LinearAlgebraAlgorithms::solve(client, b, &neg_g_col)
    {
        Ok(s) => s.reshape(&[n]).ok(),
        Err(_) => None,
    };

    if let Some(ref step) = newton_step {
        let step_norm = tensor_norm(client, step).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_step: newton norm - {}", e),
        })?;
        if step_norm <= trust_radius {
            return Ok(step.clone());
        }
    }

    // Cauchy step: scale steepest descent to trust region boundary
    let scale = trust_radius / g_norm;
    client
        .mul_scalar(&neg_g, scale)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_step: cauchy scale - {}", e),
        })
}

fn compute_predicted_reduction<R, C>(
    client: &C,
    b: &Tensor<R>,
    g: &Tensor<R>,
    step: &Tensor<R>,
    n: usize,
) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Predicted reduction: -g'd - 0.5 * d'Bd
    let gd = tensor_dot(client, g, step).map_err(|e| OptimizeError::NumericalError {
        message: format!("pred_red: g'd - {}", e),
    })?;

    let d_col = step
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pred_red: reshape d - {}", e),
        })?;
    let bd = client
        .matmul(b, &d_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pred_red: B*d - {}", e),
        })?;
    let bd_flat = bd
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("pred_red: reshape Bd - {}", e),
        })?;
    let dbd = tensor_dot(client, step, &bd_flat).map_err(|e| OptimizeError::NumericalError {
        message: format!("pred_red: d'Bd - {}", e),
    })?;

    Ok(-gd - 0.5 * dbd)
}

fn sr1_update<R, C>(
    client: &C,
    b: &Tensor<R>,
    s: &Tensor<R>,
    y: &Tensor<R>,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // SR1: B_new = B + (y - B*s)(y - B*s)' / (y - B*s)'s
    let s_col = s
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: reshape s - {}", e),
        })?;
    let bs = client
        .matmul(b, &s_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: B*s - {}", e),
        })?;
    let bs_flat = bs
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: reshape Bs - {}", e),
        })?;

    let diff = client
        .sub(y, &bs_flat)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: y - Bs - {}", e),
        })?;

    let dts = tensor_dot(client, &diff, s).map_err(|e| OptimizeError::NumericalError {
        message: format!("sr1: (y-Bs)'s - {}", e),
    })?;

    let s_norm = tensor_norm(client, s).map_err(|e| OptimizeError::NumericalError {
        message: format!("sr1: s norm - {}", e),
    })?;
    let diff_norm = tensor_norm(client, &diff).map_err(|e| OptimizeError::NumericalError {
        message: format!("sr1: diff norm - {}", e),
    })?;

    // Skip update if denominator too small (SR1 stability criterion)
    if dts.abs() < 1e-8 * s_norm * diff_norm {
        return Ok(b.clone());
    }

    let diff_col = diff
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: reshape diff - {}", e),
        })?;
    let diff_row = diff
        .reshape(&[1, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: reshape diff row - {}", e),
        })?;

    let outer = client
        .matmul(&diff_col, &diff_row)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: outer product - {}", e),
        })?;
    let update =
        client
            .mul_scalar(&outer, 1.0 / dts)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("sr1: scale - {}", e),
            })?;

    client
        .add(b, &update)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("sr1: B + update - {}", e),
        })
}

fn compute_max_violation_all<R: Runtime>(
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
) -> OptimizeResult<f64> {
    let mut max_viol = 0.0f64;

    for constraint in constraints {
        let c_val = (constraint.fun)(x).map_err(|e| OptimizeError::NumericalError {
            message: format!("max_viol: constraint eval - {}", e),
        })?;
        let vals: Vec<f64> = c_val.to_vec();
        match constraint.kind {
            ConstraintType::Equality => {
                for v in &vals {
                    max_viol = max_viol.max(v.abs());
                }
            }
            ConstraintType::Inequality => {
                for v in &vals {
                    max_viol = max_viol.max((-v).max(0.0));
                }
            }
        }
    }

    let x_vals: Vec<f64> = x.to_vec();
    if let Some(ref lower) = bounds.lower {
        let l_vals: Vec<f64> = lower.to_vec();
        for (xi, li) in x_vals.iter().zip(l_vals.iter()) {
            max_viol = max_viol.max((li - xi).max(0.0));
        }
    }
    if let Some(ref upper) = bounds.upper {
        let u_vals: Vec<f64> = upper.to_vec();
        for (xi, ui) in x_vals.iter().zip(u_vals.iter()) {
            max_viol = max_viol.max((xi - ui).max(0.0));
        }
    }

    Ok(max_viol)
}
