//! SLSQP (Sequential Least Squares Programming) implementation.
//!
//! Tensor-based SQP algorithm with BFGS Hessian approximation.
//! Reference: scipy.optimize.slsqp

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
    finite_difference_gradient, tensor_dot, tensor_norm,
};

use super::qp_subproblem::qp_subproblem_impl;

/// SLSQP implementation generic over Runtime.
pub fn slsqp_impl<R, C, F>(
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
            context: "slsqp: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();

    // Apply bounds to initial point
    x = apply_bounds(client, &x, bounds)?;

    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("slsqp: initial f eval - {}", e),
    })?;
    let mut nfev = 1usize;

    // Initialize BFGS Hessian approximation as identity
    let mut b_hess =
        client
            .eye(n, None, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: create identity - {}", e),
            })?;

    // Separate constraints by type
    let eq_indices: Vec<usize> = constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.kind == ConstraintType::Equality)
        .map(|(i, _)| i)
        .collect();
    let ineq_indices: Vec<usize> = constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.kind == ConstraintType::Inequality)
        .map(|(i, _)| i)
        .collect();

    for iter in 0..options.max_iter {
        // Compute objective gradient via finite differences
        let grad = finite_difference_gradient(client, &f, &x, fx, options.eps).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("slsqp: gradient - {}", e),
            }
        })?;
        nfev += n;

        // Evaluate constraints and build Jacobians
        let (a_eq, c_eq) = evaluate_constraints(
            client,
            &x,
            constraints,
            &eq_indices,
            n,
            options.eps,
            &mut nfev,
        )?;
        let (a_ineq, c_ineq) = evaluate_constraints(
            client,
            &x,
            constraints,
            &ineq_indices,
            n,
            options.eps,
            &mut nfev,
        )?;

        // Add bound constraints as linear inequality constraints
        let (a_ineq_full, c_ineq_full) =
            add_bound_constraints(client, &x, bounds, a_ineq.as_ref(), c_ineq.as_ref(), n)?;

        // Check KKT convergence
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("slsqp: grad norm - {}", e),
        })?;

        let max_violation = compute_max_violation(client, c_eq.as_ref(), c_ineq_full.as_ref())?;

        if grad_norm < options.tol && max_violation < options.constraint_tol {
            return Ok(ConstrainedResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
                constraint_violation: max_violation,
                message: "Optimization converged".to_string(),
            });
        }

        // Solve QP subproblem
        let qp_result = qp_subproblem_impl(
            client,
            &b_hess,
            &grad,
            a_eq.as_ref(),
            c_eq.as_ref(),
            a_ineq_full.as_ref(),
            c_ineq_full.as_ref(),
        )?;

        let d = &qp_result.d;

        // L1 merit function line search
        let (x_new, fx_new, ls_evals) =
            merit_line_search(client, &f, &x, d, fx, constraints, options)?;
        nfev += ls_evals;

        // Compute step and gradient differences for BFGS update
        let s = client
            .sub(&x_new, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: s = x_new - x - {}", e),
            })?;

        let s_norm = tensor_norm(client, &s).map_err(|e| OptimizeError::NumericalError {
            message: format!("slsqp: s norm - {}", e),
        })?;

        if s_norm < 1e-16 {
            return Ok(ConstrainedResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
                constraint_violation: max_violation,
                message: "Converged (step size below threshold)".to_string(),
            });
        }

        let grad_new = finite_difference_gradient(client, &f, &x_new, fx_new, options.eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: new gradient - {}", e),
            })?;
        nfev += n;

        let y = client
            .sub(&grad_new, &grad)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: y = grad_new - grad - {}", e),
            })?;

        // Damped BFGS update of Lagrangian Hessian
        b_hess = bfgs_update(client, &b_hess, &s, &y, n)?;

        x = apply_bounds(client, &x_new, bounds)?;
        fx = fx_new;
    }

    let max_violation = {
        let (_, c_eq_final) = evaluate_constraints(
            client,
            &x,
            constraints,
            &eq_indices,
            n,
            options.eps,
            &mut nfev,
        )?;
        let (_, c_ineq_final) = evaluate_constraints(
            client,
            &x,
            constraints,
            &ineq_indices,
            n,
            options.eps,
            &mut nfev,
        )?;
        compute_max_violation(client, c_eq_final.as_ref(), c_ineq_final.as_ref())?
    };

    Ok(ConstrainedResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
        constraint_violation: max_violation,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Evaluate constraints and compute Jacobians.
///
/// Returns (Jacobian [m, n], constraint_values [m]) or (None, None) if no constraints.
#[allow(clippy::type_complexity)]
fn evaluate_constraints<R, C>(
    client: &C,
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    indices: &[usize],
    n: usize,
    eps: f64,
    nfev: &mut usize,
) -> OptimizeResult<(Option<Tensor<R>>, Option<Tensor<R>>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    if indices.is_empty() {
        return Ok((None, None));
    }

    let mut c_parts: Vec<Tensor<R>> = Vec::new();
    let mut j_parts: Vec<Tensor<R>> = Vec::new();

    for &idx in indices {
        let constraint = &constraints[idx];

        // Evaluate constraint
        let c_val = (constraint.fun)(x).map_err(|e| OptimizeError::NumericalError {
            message: format!("slsqp: constraint {} eval - {}", idx, e),
        })?;
        *nfev += 1;

        let m_i = c_val.shape()[0];

        // Compute or use provided Jacobian
        let jac = if let Some(jac_fn) = constraint.jac {
            let j = jac_fn(x).map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: constraint {} jac - {}", idx, e),
            })?;
            *nfev += 1;
            j
        } else {
            // Finite difference Jacobian
            let j = crate::optimize::minimize::impl_generic::utils::finite_difference_jacobian(
                client,
                &|x_inner: &Tensor<R>| (constraint.fun)(x_inner),
                x,
                &c_val,
                m_i,
                n,
                eps,
            )
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("slsqp: constraint {} fd jac - {}", idx, e),
            })?;
            *nfev += n;
            j
        };

        c_parts.push(c_val);
        j_parts.push(jac);
    }

    // Concatenate all constraint values and Jacobians
    let c_refs: Vec<&Tensor<R>> = c_parts.iter().collect();
    let j_refs: Vec<&Tensor<R>> = j_parts.iter().collect();

    let c_all = client
        .cat(&c_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("slsqp: cat constraint vals - {}", e),
        })?;
    let j_all = client
        .cat(&j_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("slsqp: cat jacobians - {}", e),
        })?;

    Ok((Some(j_all), Some(c_all)))
}

/// Add bound constraints as linear inequality constraints.
///
/// Lower bounds: x_i - l_i >= 0  ->  e_i'*d >= -(x_i - l_i)
/// Upper bounds: u_i - x_i >= 0  ->  -e_i'*d >= -(u_i - x_i)
#[allow(clippy::type_complexity)]
fn add_bound_constraints<R, C>(
    client: &C,
    x: &Tensor<R>,
    bounds: &Bounds<R>,
    a_ineq: Option<&Tensor<R>>,
    c_ineq: Option<&Tensor<R>>,
    n: usize,
) -> OptimizeResult<(Option<Tensor<R>>, Option<Tensor<R>>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut a_parts: Vec<Tensor<R>> = Vec::new();
    let mut c_parts: Vec<Tensor<R>> = Vec::new();

    if let Some(a) = a_ineq {
        a_parts.push(a.clone());
    }
    if let Some(c) = c_ineq {
        c_parts.push(c.clone());
    }

    // Lower bounds: x - lower >= 0
    if let Some(ref lower) = bounds.lower {
        let identity =
            client
                .eye(n, None, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bounds: identity - {}", e),
                })?;
        let c_lower = client
            .sub(x, lower)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bounds: x - lower - {}", e),
            })?;
        a_parts.push(identity);
        c_parts.push(c_lower);
    }

    // Upper bounds: upper - x >= 0
    if let Some(ref upper) = bounds.upper {
        let neg_identity =
            client
                .eye(n, None, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bounds: identity - {}", e),
                })?;
        let neg_identity =
            client
                .mul_scalar(&neg_identity, -1.0)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bounds: negate identity - {}", e),
                })?;
        let c_upper = client
            .sub(upper, x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bounds: upper - x - {}", e),
            })?;
        a_parts.push(neg_identity);
        c_parts.push(c_upper);
    }

    if a_parts.is_empty() {
        return Ok((None, None));
    }

    let a_refs: Vec<&Tensor<R>> = a_parts.iter().collect();
    let c_refs: Vec<&Tensor<R>> = c_parts.iter().collect();

    let a_full = client
        .cat(&a_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bounds: cat A - {}", e),
        })?;
    let c_full = client
        .cat(&c_refs, 0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bounds: cat c - {}", e),
        })?;

    Ok((Some(a_full), Some(c_full)))
}

/// Compute maximum constraint violation.
fn compute_max_violation<R, C>(
    client: &C,
    c_eq: Option<&Tensor<R>>,
    c_ineq: Option<&Tensor<R>>,
) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut max_viol = 0.0f64;

    if let Some(c) = c_eq {
        // Max of absolute values for equality constraints
        let abs_c = client.abs(c).map_err(|e| OptimizeError::NumericalError {
            message: format!("compute_max_violation: abs eq - {}", e),
        })?;
        let max_eq =
            client
                .max(&abs_c, &[0], false)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("compute_max_violation: max eq - {}", e),
                })?;
        max_viol = max_viol.max(max_eq.to_vec()[0]); // Single scalar extraction
    }

    if let Some(c) = c_ineq {
        // For inequality: c >= 0, violation = max(0, -c)
        let neg_c = client
            .mul_scalar(c, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("compute_max_violation: negate ineq - {}", e),
            })?;
        let zeros =
            client
                .fill(c.shape(), 0.0, c.dtype())
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("compute_max_violation: zeros - {}", e),
                })?;
        let violations =
            client
                .maximum(&neg_c, &zeros)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("compute_max_violation: max(0, -c) - {}", e),
                })?;
        let max_ineq =
            client
                .max(&violations, &[0], false)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("compute_max_violation: max ineq - {}", e),
                })?;
        max_viol = max_viol.max(max_ineq.to_vec()[0]); // Single scalar extraction
    }

    Ok(max_viol)
}

/// L1 merit function line search.
fn merit_line_search<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    d: &Tensor<R>,
    fx: f64,
    constraints: &[Constraint<'_, R>],
    _options: &ConstrainedOptions,
) -> OptimizeResult<(Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mu = 1.0; // L1 penalty parameter
    let mut alpha = 1.0;
    let rho = 0.5;
    let c_armijo = 0.0001;
    let mut nfev = 0usize;

    // Current merit value
    let current_merit = compute_merit(client, f, x, constraints, fx, mu)?;
    nfev += constraints.len();

    for _ in 0..30 {
        let scaled_d = client
            .mul_scalar(d, alpha)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("merit_ls: scale d - {}", e),
            })?;
        let x_new = client
            .add(x, &scaled_d)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("merit_ls: x + alpha*d - {}", e),
            })?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("merit_ls: f eval - {}", e),
        })?;
        nfev += 1;

        let new_merit = compute_merit(client, f, &x_new, constraints, fx_new, mu)?;
        nfev += constraints.len();

        if new_merit <= current_merit - c_armijo * alpha * current_merit.abs().max(1.0) {
            return Ok((x_new, fx_new, nfev));
        }

        alpha *= rho;
    }

    // Fallback: return with smallest step
    let scaled_d = client
        .mul_scalar(d, alpha)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("merit_ls: fallback scale - {}", e),
        })?;
    let x_new = client
        .add(x, &scaled_d)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("merit_ls: fallback add - {}", e),
        })?;
    let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
        message: format!("merit_ls: fallback f - {}", e),
    })?;
    nfev += 1;

    Ok((x_new, fx_new, nfev))
}

/// Compute L1 merit function: f(x) + mu * sum(|c_eq|) + mu * sum(max(0, -c_ineq))
fn compute_merit<R, C, F>(
    client: &C,
    _f: &F,
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    fx: f64,
    mu: f64,
) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut penalty = 0.0;

    for constraint in constraints {
        let c_val = (constraint.fun)(x).map_err(|e| OptimizeError::NumericalError {
            message: format!("merit: constraint eval - {}", e),
        })?;

        match constraint.kind {
            ConstraintType::Equality => {
                // Sum of absolute values
                let abs_c = client
                    .abs(&c_val)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("merit: abs eq - {}", e),
                    })?;
                let sum_c =
                    client
                        .sum(&abs_c, &[0], false)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("merit: sum abs eq - {}", e),
                        })?;
                let sc: f64 = sum_c.to_vec()[0]; // Single scalar extraction
                penalty += sc;
            }
            ConstraintType::Inequality => {
                // Sum of max(0, -c)
                let neg_c =
                    client
                        .mul_scalar(&c_val, -1.0)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("merit: negate ineq - {}", e),
                        })?;
                let zeros = client
                    .fill(c_val.shape(), 0.0, c_val.dtype())
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("merit: zeros - {}", e),
                    })?;
                let violations =
                    client
                        .maximum(&neg_c, &zeros)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("merit: max(0, -c) - {}", e),
                        })?;
                let sum_viol = client.sum(&violations, &[0], false).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("merit: sum ineq - {}", e),
                    }
                })?;
                let sv: f64 = sum_viol.to_vec()[0]; // Single scalar extraction
                penalty += sv;
            }
        }
    }

    Ok(fx + mu * penalty)
}

/// Damped BFGS update of Hessian approximation.
fn bfgs_update<R, C>(
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
    let sy = tensor_dot(client, s, y).map_err(|e| OptimizeError::NumericalError {
        message: format!("bfgs_update: s'y - {}", e),
    })?;

    // Compute B*s
    let s_col = s
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: reshape s - {}", e),
        })?;
    let bs = client
        .matmul(b, &s_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: B*s - {}", e),
        })?;
    let bs_flat = bs
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: reshape B*s - {}", e),
        })?;
    let sbs = tensor_dot(client, s, &bs_flat).map_err(|e| OptimizeError::NumericalError {
        message: format!("bfgs_update: s'B*s - {}", e),
    })?;

    if sbs.abs() < 1e-16 {
        return Ok(b.clone());
    }

    // Powell's damping for Hessian update
    let theta = if sy >= 0.2 * sbs {
        1.0
    } else {
        0.8 * sbs / (sbs - sy)
    };

    // r = theta * y + (1 - theta) * B*s
    let r = if (theta - 1.0).abs() < 1e-15 {
        y.clone()
    } else {
        let term1 = client
            .mul_scalar(y, theta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs_update: theta*y - {}", e),
            })?;
        let term2 = client.mul_scalar(&bs_flat, 1.0 - theta).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("bfgs_update: (1-theta)*Bs - {}", e),
            }
        })?;
        client
            .add(&term1, &term2)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs_update: r = theta*y + ... - {}", e),
            })?
    };

    let sr = tensor_dot(client, s, &r).map_err(|e| OptimizeError::NumericalError {
        message: format!("bfgs_update: s'r - {}", e),
    })?;

    if sr.abs() < 1e-16 {
        return Ok(b.clone());
    }

    // B_new = B - (B*s)(B*s)' / (s'B*s) + r*r' / (s'r)
    let s_row = s
        .reshape(&[1, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: reshape s row - {}", e),
        })?;
    let r_col = r
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: reshape r col - {}", e),
        })?;
    let r_row = r
        .reshape(&[1, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: reshape r row - {}", e),
        })?;

    // (B*s)(B*s)' / sbs
    let bs_bst = client
        .matmul(
            &bs,
            &client
                .matmul(&s_row, b)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("bfgs_update: s'B - {}", e),
                })?,
        )
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: Bs*s'B - {}", e),
        })?;
    let term1 =
        client
            .mul_scalar(&bs_bst, 1.0 / sbs)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs_update: Bs*s'B/sbs - {}", e),
            })?;

    // r*r' / sr
    let rrt = client
        .matmul(&r_col, &r_row)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: r*r' - {}", e),
        })?;
    let term2 = client
        .mul_scalar(&rrt, 1.0 / sr)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: r*r'/sr - {}", e),
        })?;

    // B_new = B - term1 + term2
    let b_minus = client
        .sub(b, &term1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: B - term1 - {}", e),
        })?;
    client
        .add(&b_minus, &term2)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs_update: B - term1 + term2 - {}", e),
        })
}

/// Apply bounds to a point by clamping.
fn apply_bounds<R, C>(client: &C, x: &Tensor<R>, bounds: &Bounds<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let mut result = x.clone();
    if let Some(ref lower) = bounds.lower {
        result = client
            .maximum(&result, lower)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("apply_bounds: max with lower - {}", e),
            })?;
    }
    if let Some(ref upper) = bounds.upper {
        result = client
            .minimum(&result, upper)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("apply_bounds: min with upper - {}", e),
            })?;
    }
    Ok(result)
}
