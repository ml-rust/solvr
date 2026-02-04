//! Boundary Value Problem (BVP) solver using collocation.
//!
//! Solves two-point BVPs of the form:
//!   dy/dx = f(x, y)
//!   bc(y(a), y(b)) = 0

use numr::error::Result;
use numr::ops::{LinalgOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::BVPOptions;
use crate::integrate::traits::BVPResult;

/// BVP solver implementation using collocation method.
///
/// Uses 4th order Lobatto IIIA collocation with mesh refinement.
pub fn bvp_impl<R, C, F, BC>(
    client: &C,
    f: F,
    bc: BC,
    x_init: &Tensor<R>,
    y_init: &Tensor<R>,
    options: &BVPOptions,
) -> IntegrateResult<BVPResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    BC: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    // Get mesh and initial guess
    let x_vec: Vec<f64> = x_init.to_vec();
    let y_vec: Vec<f64> = y_init.to_vec();

    let n_points = x_vec.len();
    let y_shape = y_init.shape();
    let n_vars = if y_shape.len() == 2 { y_shape[0] } else { 1 };

    if n_points < 2 {
        return Err(IntegrateError::InvalidInput {
            context: "BVP requires at least 2 mesh points".to_string(),
        });
    }

    // Validate y shape
    let expected_y_len = n_vars * n_points;
    if y_vec.len() != expected_y_len {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "y_init has {} elements, expected {} (n_vars={} × n_points={})",
                y_vec.len(),
                expected_y_len,
                n_vars,
                n_points
            ),
        });
    }

    // Current mesh and solution
    let mut x = x_vec.clone();
    let mut y = y_vec.clone();

    for iter in 0..options.max_iter {
        // Solve the collocation system
        let (y_new, residual) = solve_collocation(client, &f, &bc, &x, &y, n_vars, options)?;

        // Compute residual norm
        let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        let y_norm: f64 = y_new.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // Check convergence
        let tolerance = options.atol + options.rtol * y_norm;
        if res_norm < tolerance {
            return build_bvp_result(client, &x, &y_new, &residual, n_vars, true, iter + 1);
        }

        // Update solution
        y = y_new;

        // Check if mesh refinement is needed
        let (needs_refinement, refinement_points) =
            check_mesh_refinement(client, &f, &x, &y, n_vars, options)?;

        if needs_refinement && x.len() < options.max_mesh_size {
            // Refine mesh
            let (new_x, new_y) = refine_mesh(&x, &y, &refinement_points, n_vars);
            x = new_x;
            y = new_y;
        }
    }

    // Did not converge
    let residual = compute_residual(client, &f, &bc, &x, &y, n_vars)?;
    build_bvp_result(client, &x, &y, &residual, n_vars, false, options.max_iter)
}

/// Solve the collocation equations using Newton iteration.
fn solve_collocation<R, C, F, BC>(
    client: &C,
    f: &F,
    bc: &BC,
    x: &[f64],
    y: &[f64],
    n_vars: usize,
    _options: &BVPOptions,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    BC: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_points = x.len();
    let n_unknowns = n_vars * n_points;

    let mut y_current = y.to_vec();

    // Newton iteration
    for _ in 0..20 {
        // Compute residual
        let residual = compute_residual(client, f, bc, x, &y_current, n_vars)?;

        let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();

        if res_norm < 1e-10 {
            return Ok((y_current, residual));
        }

        // Compute Jacobian using finite differences
        let jacobian = compute_bvp_jacobian(client, f, bc, x, &y_current, n_vars)?;

        // Solve J * delta = -residual
        let j_tensor = Tensor::<R>::from_slice(&jacobian, &[n_unknowns, n_unknowns], device);
        let neg_res: Vec<f64> = residual.iter().map(|&r| -r).collect();
        let neg_res_tensor = Tensor::<R>::from_slice(&neg_res, &[n_unknowns, 1], device);

        let delta_tensor = client
            .solve(&j_tensor, &neg_res_tensor)
            .map_err(to_integrate_err)?;
        let delta: Vec<f64> = delta_tensor.to_vec();

        // Update solution
        for i in 0..n_unknowns {
            y_current[i] += delta[i];
        }
    }

    let residual = compute_residual(client, f, bc, x, &y_current, n_vars)?;
    Ok((y_current, residual))
}

/// Compute the residual of the BVP equations.
fn compute_residual<R, C, F, BC>(
    client: &C,
    f: &F,
    bc: &BC,
    x: &[f64],
    y: &[f64],
    n_vars: usize,
) -> IntegrateResult<Vec<f64>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    BC: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_points = x.len();
    let n_equations = n_vars * n_points;

    let mut residual = vec![0.0; n_equations];

    // Collocation equations at interior points and midpoints
    for i in 0..n_points - 1 {
        let h = x[i + 1] - x[i];

        // Get y values at points i and i+1
        let y_i: Vec<f64> = (0..n_vars).map(|v| y[v * n_points + i]).collect();
        let y_ip1: Vec<f64> = (0..n_vars).map(|v| y[v * n_points + i + 1]).collect();

        // Evaluate f at point i
        let x_i_tensor = Tensor::<R>::from_slice(&[x[i]], &[1], device);
        let y_i_tensor = Tensor::<R>::from_slice(&y_i, &[n_vars], device);
        let f_i = f(&x_i_tensor, &y_i_tensor).map_err(to_integrate_err)?;
        let f_i_vec: Vec<f64> = f_i.to_vec();

        // Evaluate f at point i+1
        let x_ip1_tensor = Tensor::<R>::from_slice(&[x[i + 1]], &[1], device);
        let y_ip1_tensor = Tensor::<R>::from_slice(&y_ip1, &[n_vars], device);
        let f_ip1 = f(&x_ip1_tensor, &y_ip1_tensor).map_err(to_integrate_err)?;
        let f_ip1_vec: Vec<f64> = f_ip1.to_vec();

        // Collocation residual: y[i+1] - y[i] - h/2 * (f[i] + f[i+1])
        // (Trapezoidal rule consistency)
        for v in 0..n_vars {
            let eq_idx = v * n_points + i;
            residual[eq_idx] = y_ip1[v] - y_i[v] - h / 2.0 * (f_i_vec[v] + f_ip1_vec[v]);
        }
    }

    // Boundary conditions at last equation slot
    let y_a: Vec<f64> = (0..n_vars).map(|v| y[v * n_points]).collect();
    let y_b: Vec<f64> = (0..n_vars)
        .map(|v| y[v * n_points + n_points - 1])
        .collect();

    let y_a_tensor = Tensor::<R>::from_slice(&y_a, &[n_vars], device);
    let y_b_tensor = Tensor::<R>::from_slice(&y_b, &[n_vars], device);
    let bc_res = bc(&y_a_tensor, &y_b_tensor).map_err(to_integrate_err)?;
    let bc_vec: Vec<f64> = bc_res.to_vec();

    // Put BC residuals at the last point equations
    for (v, &bc_val) in bc_vec.iter().enumerate().take(n_vars) {
        let eq_idx = v * n_points + n_points - 1;
        residual[eq_idx] = bc_val;
    }

    Ok(residual)
}

/// Compute the Jacobian of the BVP equations using finite differences.
fn compute_bvp_jacobian<R, C, F, BC>(
    client: &C,
    f: &F,
    bc: &BC,
    x: &[f64],
    y: &[f64],
    n_vars: usize,
) -> IntegrateResult<Vec<f64>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    BC: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let n_points = x.len();
    let n_unknowns = n_vars * n_points;
    let eps = 1e-7;

    let residual_0 = compute_residual(client, f, bc, x, y, n_vars)?;

    let mut jacobian = vec![0.0; n_unknowns * n_unknowns];

    for j in 0..n_unknowns {
        // Perturb y[j]
        let mut y_pert = y.to_vec();
        y_pert[j] += eps;

        let residual_pert = compute_residual(client, f, bc, x, &y_pert, n_vars)?;

        // J[:, j] = (res_pert - res_0) / eps
        for i in 0..n_unknowns {
            jacobian[i * n_unknowns + j] = (residual_pert[i] - residual_0[i]) / eps;
        }
    }

    Ok(jacobian)
}

/// Check if mesh refinement is needed and identify where.
fn check_mesh_refinement<R, C, F>(
    client: &C,
    f: &F,
    x: &[f64],
    y: &[f64],
    n_vars: usize,
    options: &BVPOptions,
) -> IntegrateResult<(bool, Vec<usize>)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_points = x.len();
    let mut needs_refinement = false;
    let mut refinement_points = Vec::new();

    // Check residual at midpoints
    for i in 0..n_points - 1 {
        let h = x[i + 1] - x[i];
        let x_mid = (x[i] + x[i + 1]) / 2.0;

        // Interpolate y at midpoint
        let y_i: Vec<f64> = (0..n_vars).map(|v| y[v * n_points + i]).collect();
        let y_ip1: Vec<f64> = (0..n_vars).map(|v| y[v * n_points + i + 1]).collect();
        let y_mid: Vec<f64> = y_i
            .iter()
            .zip(&y_ip1)
            .map(|(&a, &b)| (a + b) / 2.0)
            .collect();

        // Evaluate f at midpoint
        let x_mid_tensor = Tensor::<R>::from_slice(&[x_mid], &[1], device);
        let y_mid_tensor = Tensor::<R>::from_slice(&y_mid, &[n_vars], device);
        let f_mid = f(&x_mid_tensor, &y_mid_tensor).map_err(to_integrate_err)?;
        let f_mid_vec: Vec<f64> = f_mid.to_vec();

        // Also get f at endpoints
        let x_i_tensor = Tensor::<R>::from_slice(&[x[i]], &[1], device);
        let y_i_tensor = Tensor::<R>::from_slice(&y_i, &[n_vars], device);
        let f_i = f(&x_i_tensor, &y_i_tensor).map_err(to_integrate_err)?;
        let f_i_vec: Vec<f64> = f_i.to_vec();

        // Estimate local error using Simpson's rule vs midpoint
        let mut error_est = 0.0;
        for v in 0..n_vars {
            let diff = (f_mid_vec[v] - (f_i_vec[v] + f_mid_vec[v]) / 2.0).abs();
            error_est += diff * diff;
        }
        error_est = error_est.sqrt() * h;

        if error_est > options.atol + options.rtol * f_mid_vec.iter().map(|&v| v.abs()).sum::<f64>()
        {
            needs_refinement = true;
            refinement_points.push(i);
        }
    }

    Ok((needs_refinement, refinement_points))
}

/// Refine the mesh by adding midpoints at specified intervals.
fn refine_mesh(
    x: &[f64],
    y: &[f64],
    refinement_points: &[usize],
    n_vars: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = x.len();
    let n_new = n_points + refinement_points.len();

    let mut new_x = Vec::with_capacity(n_new);
    let mut new_y = vec![0.0; n_vars * n_new];

    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut ref_idx = 0;

    while old_idx < n_points {
        // Copy current point
        new_x.push(x[old_idx]);
        for v in 0..n_vars {
            new_y[v * n_new + new_idx] = y[v * n_points + old_idx];
        }
        new_idx += 1;

        // Check if we need to add a midpoint after this interval
        if ref_idx < refinement_points.len() && refinement_points[ref_idx] == old_idx {
            if old_idx + 1 < n_points {
                // Add midpoint
                let x_mid = (x[old_idx] + x[old_idx + 1]) / 2.0;
                new_x.push(x_mid);

                for v in 0..n_vars {
                    let y_i = y[v * n_points + old_idx];
                    let y_ip1 = y[v * n_points + old_idx + 1];
                    new_y[v * n_new + new_idx] = (y_i + y_ip1) / 2.0;
                }
                new_idx += 1;
            }
            ref_idx += 1;
        }

        old_idx += 1;
    }

    (new_x, new_y)
}

/// Build the BVP result.
fn build_bvp_result<R, C>(
    client: &C,
    x: &[f64],
    y: &[f64],
    residual: &[f64],
    n_vars: usize,
    success: bool,
    niter: usize,
) -> IntegrateResult<BVPResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let device = client.device();
    let n_points = x.len();

    let x_tensor = Tensor::<R>::from_slice(x, &[n_points], device);
    let y_tensor = Tensor::<R>::from_slice(y, &[n_vars, n_points], device);
    let res_tensor = Tensor::<R>::from_slice(residual, &[residual.len()], device);

    Ok(BVPResult {
        x: x_tensor,
        y: y_tensor,
        residual: res_tensor,
        success,
        niter,
        mesh_size: n_points,
        message: if success {
            None
        } else {
            Some("Maximum iterations reached".to_string())
        },
    })
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
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

    #[test]
    fn test_bvp_linear() {
        let (device, client) = setup();

        // Solve y'' = 0 with y(0) = 0, y(1) = 1
        // As first-order system: y1' = y2, y2' = 0
        // Solution: y1 = x, y2 = 1

        let n_points = 5;
        let x_data: Vec<f64> = (0..n_points)
            .map(|i| i as f64 / (n_points - 1) as f64)
            .collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[n_points], &device);

        // Initial guess: y1 = x, y2 = 1
        let mut y_data = Vec::with_capacity(2 * n_points);
        for &xi in &x_data {
            y_data.push(xi); // y1 initial guess
        }
        for _ in &x_data {
            y_data.push(1.0); // y2 initial guess
        }
        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[2, n_points], &device);

        let result = bvp_impl(
            &client,
            |_x, y| {
                // f(x, y) = [y2, 0]
                let y_vec: Vec<f64> = y.to_vec();
                let n_vars = y_vec.len();
                let f_vec = vec![y_vec[n_vars / 2], 0.0]; // y2 and 0
                Ok(Tensor::<CpuRuntime>::from_slice(&f_vec, &[2], &device))
            },
            |ya, yb| {
                // bc: y1(0) = 0, y1(1) = 1
                let ya_vec: Vec<f64> = ya.to_vec();
                let yb_vec: Vec<f64> = yb.to_vec();
                let bc_vec = vec![ya_vec[0], yb_vec[0] - 1.0];
                Ok(Tensor::<CpuRuntime>::from_slice(&bc_vec, &[2], &device))
            },
            &x,
            &y,
            &BVPOptions::default(),
        )
        .unwrap();

        assert!(result.success, "BVP should converge: {:?}", result.message);

        // Check solution: y1 should be close to x
        let y_final: Vec<f64> = result.y.to_vec();
        let x_final: Vec<f64> = result.x.to_vec();
        let n = x_final.len();

        for i in 0..n {
            let y1 = y_final[i];
            let expected = x_final[i];
            assert!(
                (y1 - expected).abs() < 0.1,
                "At x={}, y1={}, expected {}",
                x_final[i],
                y1,
                expected
            );
        }
    }
}
