//! Consistent initial condition computation for DAE solvers.
//!
//! For a DAE F(t, y, y') = 0, user-provided (y0, yp0) may not satisfy
//! F(t0, y0, yp0) = 0. This module provides Newton iteration to refine
//! the initial conditions to consistency.
//!
//! # Strategy (IDA-style)
//!
//! Given variable classifications:
//! - Fix differential variables in y0 (user-specified values)
//! - Solve for: algebraic variables in y0 AND all components of yp0
//!
//! Without classifications, we fix all of y0 and solve only for yp0.

use numr::autograd::DualTensor;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{LinalgOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::{DAEOptions, DAEVariableType};

use super::dae_jacobian::eval_dae_primal;
use super::jacobian::compute_norm_scalar;

/// Refine initial conditions to satisfy F(t0, y0, yp0) ≈ 0.
///
/// Uses Newton iteration to find consistent (y0, yp0) starting from the
/// user's guess. The strategy depends on variable type classifications:
///
/// - With classifications: Fix differential vars in y, solve for algebraic y and all yp
/// - Without classifications: Fix all y, solve only for yp
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - DAE residual function
/// * `t0` - Initial time
/// * `y0` - Initial guess for state
/// * `yp0` - Initial guess for derivative
/// * `dae_options` - Options including variable types and IC tolerance
///
/// # Returns
///
/// Tuple of (refined_y0, refined_yp0, number_of_iterations)
pub fn compute_consistent_ic<R, C, F>(
    client: &C,
    f: &F,
    t0: f64,
    y0: &Tensor<R>,
    yp0: &Tensor<R>,
    dae_options: &DAEOptions<R>,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let n = y0.shape()[0];
    let t = Tensor::<R>::from_slice(&[t0], &[1], device);

    // Check initial residual
    let residual = eval_dae_primal(client, f, &t, y0, yp0).map_err(to_integrate_err)?;
    let res_norm = compute_norm_scalar(client, &residual, 2.0).map_err(to_integrate_err)?;

    // If already consistent, return immediately
    if res_norm < dae_options.ic_tol {
        return Ok((y0.clone(), yp0.clone(), 0));
    }

    // Determine which variables to solve for
    // Without variable types: fix y, solve for yp
    // With variable types: fix differential y, solve for algebraic y and all yp
    // If all variables are differential, just use yp-only path
    let (fix_y_mask, solve_for_yp_only) = match &dae_options.variable_types {
        Some(types) => {
            let has_algebraic = types.contains(&DAEVariableType::Algebraic);
            if !has_algebraic {
                // All differential - use simple yp-only path
                (None, true)
            } else {
                let mask: Vec<f64> = types
                    .iter()
                    .map(|t| {
                        if *t == DAEVariableType::Differential {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                (Some(Tensor::<R>::from_slice(&mask, &[n], device)), false)
            }
        }
        None => (None, true),
    };

    let mut y = y0.clone();
    let mut yp = yp0.clone();

    for iter in 0..dae_options.max_ic_iter {
        // Evaluate residual
        let residual = eval_dae_primal(client, f, &t, &y, &yp).map_err(to_integrate_err)?;
        let res_norm = compute_norm_scalar(client, &residual, 2.0).map_err(to_integrate_err)?;

        if res_norm < dae_options.ic_tol {
            return Ok((y, yp, iter + 1));
        }

        if solve_for_yp_only {
            // Simple case: only solve for yp
            // Newton: yp_new = yp - J_yp^{-1} * F
            // Compute J_yp = ∂F/∂y'
            let j_yp = compute_ic_jacobian_yp(client, f, &t, &y, &yp).map_err(to_integrate_err)?;

            // Solve J_yp * delta = -residual
            let neg_res = client
                .mul_scalar(&residual, -1.0)
                .map_err(to_integrate_err)?;
            let neg_res_col = neg_res.reshape(&[n, 1]).map_err(to_integrate_err)?;
            let delta_col = client
                .solve(&j_yp, &neg_res_col)
                .map_err(to_integrate_err)?;
            let delta = delta_col.reshape(&[n]).map_err(to_integrate_err)?;

            yp = client.add(&yp, &delta).map_err(to_integrate_err)?;
        } else {
            // General case: solve for algebraic y and all yp
            // Build combined system with 2n unknowns: [y_alg, yp]
            // We solve for delta_y_alg and delta_yp together

            // For simplicity, use a sequential approach:
            // 1. Fix differential y, compute J_yp
            // 2. Compute contribution from algebraic y using J_y_alg
            // This is a simplified Newton that works for index-1 DAEs

            // Full Newton would require forming a larger Jacobian, but for IC
            // computation we can iterate between y_alg and yp updates

            // Compute full Jacobians
            let (j_y, j_yp) =
                compute_ic_jacobians(client, f, &t, &y, &yp).map_err(to_integrate_err)?;

            // For index-1 DAEs, J_yp is typically non-singular for differential vars
            // and J_y for algebraic vars. We solve the combined system:
            // [J_y J_yp] [delta_y; delta_yp] = -F

            // Simplified approach: use J = J_y + J_yp and solve for combined delta
            // then apply mask to separate y and yp updates
            let j_combined = client.add(&j_y, &j_yp).map_err(to_integrate_err)?;

            let neg_res = client
                .mul_scalar(&residual, -1.0)
                .map_err(to_integrate_err)?;
            let neg_res_col = neg_res.reshape(&[n, 1]).map_err(to_integrate_err)?;
            let delta_col = client
                .solve(&j_combined, &neg_res_col)
                .map_err(to_integrate_err)?;
            let delta = delta_col.reshape(&[n]).map_err(to_integrate_err)?;

            // Apply updates based on variable types
            // Differential y: no change, Algebraic y: update
            // All yp: update
            if let Some(ref mask) = fix_y_mask {
                // delta_y_alg = (1 - mask) * delta (algebraic vars)
                let ones = Tensor::<R>::ones(&[n], DType::F64, client.device());
                let inv_mask = client.sub(&ones, mask).map_err(to_integrate_err)?;
                let delta_y = client.mul(&inv_mask, &delta).map_err(to_integrate_err)?;
                y = client.add(&y, &delta_y).map_err(to_integrate_err)?;
            }

            // Update yp (always)
            yp = client.add(&yp, &delta).map_err(to_integrate_err)?;
        }
    }

    // Failed to converge
    let final_residual = eval_dae_primal(client, f, &t, &y, &yp).map_err(to_integrate_err)?;
    let final_norm = compute_norm_scalar(client, &final_residual, 2.0).map_err(to_integrate_err)?;

    Err(IntegrateError::InconsistentInitialConditions {
        residual_norm: final_norm,
        tolerance: dae_options.ic_tol,
        iterations: dae_options.max_ic_iter,
    })
}

/// Compute ∂F/∂y' for IC computation.
fn compute_ic_jacobian_yp<R, C, F>(
    client: &C,
    f: &F,
    t: &Tensor<R>,
    y: &Tensor<R>,
    yp: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    use crate::common::jacobian::jacobian_autograd;

    let t_dual = DualTensor::new(t.clone(), None);
    let y_const = DualTensor::new(y.clone(), None);

    jacobian_autograd(client, |yp_dual, c| f(&t_dual, &y_const, yp_dual, c), yp)
}

/// Compute both ∂F/∂y and ∂F/∂y' for IC computation.
fn compute_ic_jacobians<R, C, F>(
    client: &C,
    f: &F,
    t: &Tensor<R>,
    y: &Tensor<R>,
    yp: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    use crate::common::jacobian::jacobian_autograd;

    let t_dual = DualTensor::new(t.clone(), None);
    let yp_const = DualTensor::new(yp.clone(), None);

    let j_y = jacobian_autograd(client, |y_dual, c| f(&t_dual, y_dual, &yp_const, c), y)?;

    let t_dual_2 = DualTensor::new(t.clone(), None);
    let y_const = DualTensor::new(y.clone(), None);

    let j_yp = jacobian_autograd(client, |yp_dual, c| f(&t_dual_2, &y_const, yp_dual, c), yp)?;

    Ok((j_y, j_yp))
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul_scalar, dual_sub};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_consistent_ic_already_consistent() {
        let (device, client) = setup();

        // F = y' - 2*y = 0, with y=1, y'=2 (consistent)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| {
            let two_y = dual_mul_scalar(y, 2.0, c)?;
            dual_sub(yp, &two_y, c)
        };

        let dae_opts = DAEOptions::<CpuRuntime>::default();
        let (y_refined, yp_refined, n_iter) =
            compute_consistent_ic(&client, &f, 0.0, &y0, &yp0, &dae_opts).unwrap();

        assert_eq!(n_iter, 0); // Already consistent
        assert!((y_refined.to_vec::<f64>()[0] - 1.0).abs() < 1e-10);
        assert!((yp_refined.to_vec::<f64>()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_consistent_ic_needs_refinement() {
        let (device, client) = setup();

        // F = y' - y = 0, with y=1, y'=0 (inconsistent, should become y'=1)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| { dual_sub(yp, y, c) };

        let dae_opts = DAEOptions::<CpuRuntime>::default();
        let (y_refined, yp_refined, n_iter) =
            compute_consistent_ic(&client, &f, 0.0, &y0, &yp0, &dae_opts).unwrap();

        assert!(n_iter > 0);
        // y should remain 1.0 (we fix y, solve for yp)
        assert!((y_refined.to_vec::<f64>()[0] - 1.0).abs() < 1e-10);
        // yp should be refined to 1.0
        assert!((yp_refined.to_vec::<f64>()[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_consistent_ic_with_variable_types() {
        let (device, client) = setup();

        // Simple DAE with variable types: F = y' - y = 0
        // With y[0] as differential, we test that variable types are respected.
        // IC: y=1, y'=0 (inconsistent), should converge to y'=1.

        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| { dual_sub(yp, y, c) };

        let dae_opts = DAEOptions::<CpuRuntime>::default()
            .with_variable_types(vec![DAEVariableType::Differential]);

        let result = compute_consistent_ic(&client, &f, 0.0, &y0, &yp0, &dae_opts);

        match result {
            Ok((y_refined, yp_refined, n_iter)) => {
                // y should remain 1.0 (fixed), yp should be refined to 1.0
                assert!(n_iter > 0);
                assert!((y_refined.to_vec::<f64>()[0] - 1.0).abs() < 1e-10);
                assert!((yp_refined.to_vec::<f64>()[0] - 1.0).abs() < 1e-8);
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}
