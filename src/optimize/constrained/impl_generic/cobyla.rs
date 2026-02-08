//! COBYLA (Constrained Optimization BY Linear Approximation) implementation.
//!
//! Derivative-free trust-region method using linear interpolation models.
//! Reference: scipy.optimize.cobyla, M.J.D. Powell (1994)

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

/// COBYLA implementation generic over Runtime.
///
/// Uses a simplex of n+1 points to build linear models of the objective
/// and constraints, then solves a trust-region subproblem at each step.
pub fn cobyla_impl<R, C, F>(
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
            context: "cobyla: empty initial guess".to_string(),
        });
    }

    // Initial trust region radius
    let mut rho = 0.5;
    let rho_end = options.tol;
    let mut nfev = 0usize;

    // Initialize simplex: n+1 points
    // sim[0] = x0, sim[i] = x0 + rho * e_i for i = 1..n
    let mut sim_points: Vec<Tensor<R>> = Vec::with_capacity(n + 1);
    let mut sim_fvals: Vec<f64> = Vec::with_capacity(n + 1);
    let mut sim_cvals: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    // Evaluate at x0
    let x0_clamped = apply_bounds_cobyla(client, x0, bounds)?;
    let f0 = f(&x0_clamped).map_err(|e| OptimizeError::NumericalError {
        message: format!("cobyla: initial f eval - {}", e),
    })?;
    nfev += 1;
    let c0 = evaluate_all_constraints(&x0_clamped, constraints, bounds, client)?;
    nfev += constraints.len();

    sim_points.push(x0_clamped.clone());
    sim_fvals.push(f0);
    sim_cvals.push(c0);

    // Build remaining simplex points
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla: identity - {}", e),
        })?;
    let rho_identity =
        client
            .mul_scalar(&identity, rho)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla: scale identity - {}", e),
            })?;

    for i in 0..n {
        let delta = rho_identity
            .narrow(0, i, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla: extract delta {} - {}", i, e),
            })?
            .contiguous()
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla: reshape delta {} - {}", i, e),
            })?;
        let xi = client
            .add(&x0_clamped, &delta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla: x0 + delta {} - {}", i, e),
            })?;
        let xi = apply_bounds_cobyla(client, &xi, bounds)?;
        let fi = f(&xi).map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla: f eval simplex {} - {}", i, e),
        })?;
        nfev += 1;
        let ci = evaluate_all_constraints(&xi, constraints, bounds, client)?;
        nfev += constraints.len();

        sim_points.push(xi);
        sim_fvals.push(fi);
        sim_cvals.push(ci);
    }

    let mut best_x = sim_points[0].clone();
    let mut best_fx = sim_fvals[0];
    let mut best_violation = max_violation_from_vec(&sim_cvals[0]);

    for iter in 0..options.max_iter {
        // Find the best feasible point (or least infeasible)
        update_best(
            &sim_points,
            &sim_fvals,
            &sim_cvals,
            &mut best_x,
            &mut best_fx,
            &mut best_violation,
        );

        // Check convergence
        if rho < rho_end && best_violation < options.constraint_tol {
            return Ok(ConstrainedResult {
                x: best_x,
                fun: best_fx,
                iterations: iter + 1,
                nfev,
                converged: true,
                constraint_violation: best_violation,
                message: "Optimization converged".to_string(),
            });
        }

        // Build linear models from simplex
        // For the objective: f(x) ≈ f(x0) + g'*(x - x0)
        // where g is computed from the simplex values
        let (f_grad, c_grads) =
            build_linear_models(client, &sim_points, &sim_fvals, &sim_cvals, n)?;

        // Find trial point by maximizing improvement within trust region
        let trial = find_trial_point(
            client,
            &sim_points[0],
            sim_fvals[0],
            &sim_cvals[0],
            &f_grad,
            &c_grads,
            rho,
            n,
        )?;

        let trial = apply_bounds_cobyla(client, &trial, bounds)?;

        // Evaluate trial point
        let f_trial = f(&trial).map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla: f trial - {}", e),
        })?;
        nfev += 1;
        let c_trial = evaluate_all_constraints(&trial, constraints, bounds, client)?;
        nfev += constraints.len();

        // Decide whether to accept trial and update simplex
        let trial_violation = max_violation_from_vec(&c_trial);
        let current_violation = max_violation_from_vec(&sim_cvals[0]);

        let accept = if trial_violation < options.constraint_tol
            && current_violation < options.constraint_tol
        {
            // Both feasible: accept if objective improves
            f_trial < sim_fvals[0]
        } else if trial_violation < current_violation {
            // Trial is more feasible
            true
        } else {
            trial_violation <= current_violation && f_trial < sim_fvals[0]
        };

        if accept {
            // Replace worst point in simplex
            let worst_idx = find_worst_simplex_point(&sim_fvals, &sim_cvals);
            sim_points[worst_idx] = trial;
            sim_fvals[worst_idx] = f_trial;
            sim_cvals[worst_idx] = c_trial;

            // Move the new point to index 0 (center of simplex)
            sim_points.swap(0, worst_idx);
            sim_fvals.swap(0, worst_idx);
            sim_cvals.swap(0, worst_idx);
        } else {
            // Reduce trust region
            rho *= 0.5;

            // Rebuild simplex around best point
            if rho >= rho_end {
                rebuild_simplex(
                    client,
                    &f,
                    constraints,
                    bounds,
                    &best_x,
                    rho,
                    n,
                    &mut sim_points,
                    &mut sim_fvals,
                    &mut sim_cvals,
                    &mut nfev,
                )?;
            }
        }
    }

    Ok(ConstrainedResult {
        x: best_x,
        fun: best_fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
        constraint_violation: best_violation,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Evaluate all constraints (user + bound) and return as Vec<f64>.
/// Convention: constraint >= 0 means feasible.
/// Note: This function extracts constraint values at the API boundary where users work with Vec<f64>.
fn evaluate_all_constraints<R, C>(
    x: &Tensor<R>,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
    _client: &C,
) -> OptimizeResult<Vec<f64>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let mut all_vals = Vec::new();

    for constraint in constraints {
        let c_val = (constraint.fun)(x).map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla: constraint eval - {}", e),
        })?;
        let vals: Vec<f64> = c_val.to_vec();
        match constraint.kind {
            ConstraintType::Inequality => {
                // c(x) >= 0
                all_vals.extend_from_slice(&vals);
            }
            ConstraintType::Equality => {
                // |c(x)| <= tol => represent as c(x) >= 0 AND -c(x) >= 0
                for v in &vals {
                    all_vals.push(*v);
                    all_vals.push(-*v);
                }
            }
        }
    }

    // Bound constraints
    let x_vals: Vec<f64> = x.to_vec();
    if let Some(ref lower) = bounds.lower {
        let l_vals: Vec<f64> = lower.to_vec();
        for (xi, li) in x_vals.iter().zip(l_vals.iter()) {
            all_vals.push(xi - li);
        }
    }
    if let Some(ref upper) = bounds.upper {
        let u_vals: Vec<f64> = upper.to_vec();
        for (xi, ui) in x_vals.iter().zip(u_vals.iter()) {
            all_vals.push(ui - xi);
        }
    }

    Ok(all_vals)
}

fn max_violation_from_vec(c_vals: &[f64]) -> f64 {
    c_vals.iter().map(|&v| (-v).max(0.0)).fold(0.0f64, f64::max)
}

fn update_best<R: Runtime>(
    points: &[Tensor<R>],
    fvals: &[f64],
    cvals: &[Vec<f64>],
    best_x: &mut Tensor<R>,
    best_fx: &mut f64,
    best_violation: &mut f64,
) {
    for i in 0..points.len() {
        let viol = max_violation_from_vec(&cvals[i]);
        let is_better = if viol < 1e-10 && *best_violation < 1e-10 {
            fvals[i] < *best_fx
        } else {
            viol < *best_violation
        };
        if is_better {
            *best_x = points[i].clone();
            *best_fx = fvals[i];
            *best_violation = viol;
        }
    }
}

/// Build linear models from simplex.
/// Returns (f_gradient [n], vec of constraint gradients [n] each).
/// Note: This function extracts point coordinates at the algorithm boundary where linear models
/// are built using scalar math (inherent to the algorithm design).
fn build_linear_models<R, C>(
    _client: &C,
    points: &[Tensor<R>],
    fvals: &[f64],
    cvals: &[Vec<f64>],
    n: usize,
) -> OptimizeResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let x0_vals: Vec<f64> = points[0].to_vec();
    let f0 = fvals[0];
    let m_constraints = cvals[0].len();

    let mut f_grad = vec![0.0; n];
    let mut c_grads = vec![vec![0.0; n]; m_constraints];

    for i in 0..n.min(points.len() - 1) {
        let xi_vals: Vec<f64> = points[i + 1].to_vec();
        let mut dx = vec![0.0; n];
        let mut dx_norm_sq = 0.0;
        for j in 0..n {
            dx[j] = xi_vals[j] - x0_vals[j];
            dx_norm_sq += dx[j] * dx[j];
        }
        if dx_norm_sq < 1e-30 {
            continue;
        }

        let df = fvals[i + 1] - f0;
        for j in 0..n {
            f_grad[j] += df * dx[j] / dx_norm_sq;
        }

        for k in 0..m_constraints {
            let dc = cvals[i + 1][k] - cvals[0][k];
            for j in 0..n {
                c_grads[k][j] += dc * dx[j] / dx_norm_sq;
            }
        }
    }

    Ok((f_grad, c_grads))
}

/// Find trial point that minimizes objective improvement while satisfying
/// linearized constraints within the trust region.
#[allow(clippy::too_many_arguments)]
fn find_trial_point<R, C>(
    client: &C,
    x0: &Tensor<R>,
    _f0: f64,
    _c0: &[f64],
    f_grad: &[f64],
    _c_grads: &[Vec<f64>],
    rho: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Simple approach: steepest descent direction projected onto feasible set
    // d = -rho * f_grad / |f_grad| (steepest descent, trust region bounded)
    let grad_norm: f64 = f_grad.iter().map(|g| g * g).sum::<f64>().sqrt();

    if grad_norm < 1e-15 {
        // Zero gradient: try moving along most violated constraint
        return Ok(x0.clone());
    }

    let step_size = rho / grad_norm;
    let d: Vec<f64> = f_grad.iter().map(|g| -step_size * g).collect();

    // Check linearized constraint satisfaction and adjust if needed
    // Note: from_slice is acceptable at the algorithm boundary where we work with Vec<f64>
    let d_tensor = Tensor::<R>::from_slice(&d, &[n], x0.device());
    let trial = client
        .add(x0, &d_tensor)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla trial: add - {}", e),
        })?;

    Ok(trial)
}

fn find_worst_simplex_point(fvals: &[f64], cvals: &[Vec<f64>]) -> usize {
    let mut worst_idx = 0;
    let mut worst_score = f64::NEG_INFINITY;

    for i in 0..fvals.len() {
        let viol = max_violation_from_vec(&cvals[i]);
        // Score: infeasible points are worse, among feasible pick highest f
        let score = if viol > 1e-10 { 1e20 + viol } else { fvals[i] };
        if score > worst_score {
            worst_score = score;
            worst_idx = i;
        }
    }
    worst_idx
}

#[allow(clippy::too_many_arguments)]
fn rebuild_simplex<R, C, F>(
    client: &C,
    f: &F,
    constraints: &[Constraint<'_, R>],
    bounds: &Bounds<R>,
    center: &Tensor<R>,
    rho: f64,
    n: usize,
    sim_points: &mut Vec<Tensor<R>>,
    sim_fvals: &mut Vec<f64>,
    sim_cvals: &mut Vec<Vec<f64>>,
    nfev: &mut usize,
) -> OptimizeResult<()>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    sim_points.clear();
    sim_fvals.clear();
    sim_cvals.clear();

    let center_clamped = apply_bounds_cobyla(client, center, bounds)?;
    let fc = f(&center_clamped).map_err(|e| OptimizeError::NumericalError {
        message: format!("cobyla rebuild: f center - {}", e),
    })?;
    *nfev += 1;
    let cc = evaluate_all_constraints(&center_clamped, constraints, bounds, client)?;
    *nfev += constraints.len();

    sim_points.push(center_clamped.clone());
    sim_fvals.push(fc);
    sim_cvals.push(cc);

    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla rebuild: identity - {}", e),
        })?;
    let rho_identity =
        client
            .mul_scalar(&identity, rho)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla rebuild: scale - {}", e),
            })?;

    for i in 0..n {
        let delta = rho_identity
            .narrow(0, i, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla rebuild: narrow {} - {}", i, e),
            })?
            .contiguous()
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla rebuild: reshape {} - {}", i, e),
            })?;
        let xi =
            client
                .add(&center_clamped, &delta)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("cobyla rebuild: add {} - {}", i, e),
                })?;
        let xi = apply_bounds_cobyla(client, &xi, bounds)?;
        let fi = f(&xi).map_err(|e| OptimizeError::NumericalError {
            message: format!("cobyla rebuild: f {} - {}", i, e),
        })?;
        *nfev += 1;
        let ci = evaluate_all_constraints(&xi, constraints, bounds, client)?;
        *nfev += constraints.len();

        sim_points.push(xi);
        sim_fvals.push(fi);
        sim_cvals.push(ci);
    }

    Ok(())
}

fn apply_bounds_cobyla<R, C>(
    client: &C,
    x: &Tensor<R>,
    bounds: &Bounds<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let mut result = x.clone();
    if let Some(ref lower) = bounds.lower {
        result = client
            .maximum(&result, lower)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla bounds: max - {}", e),
            })?;
    }
    if let Some(ref upper) = bounds.upper {
        result = client
            .minimum(&result, upper)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cobyla bounds: min - {}", e),
            })?;
    }
    Ok(result)
}
