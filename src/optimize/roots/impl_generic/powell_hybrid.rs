//! Powell hybrid method for systems of nonlinear equations.
//!
//! Combines Newton steps with a dogleg trust region fallback.
//! Uses finite-difference Jacobian initially, then applies Broyden rank-1
//! updates between full Jacobian recalculations.
//!
//! # Algorithm
//!
//! 1. Compute Jacobian J via finite differences
//! 2. Compute Newton step: p_N = -J^{-1} F(x)
//! 3. Compute steepest descent step: p_SD = -J^T F / ||J^T F||^2
//! 4. Use dogleg combination within trust region
//! 5. Update trust region based on actual vs predicted reduction
//! 6. Apply Broyden rank-1 update to Jacobian (avoid full FD recomputation)
use crate::DType;

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::utils::{finite_difference_jacobian, tensor_dot, tensor_norm};
use crate::optimize::roots::RootOptions;

use super::TensorRootResult;

/// Powell hybrid method implementation.
pub fn powell_hybrid_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &RootOptions,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "powell_hybrid: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("powell_hybrid: initial eval - {}", e),
    })?;

    // Initial Jacobian via finite differences
    let mut jac =
        finite_difference_jacobian(client, &f, &x, &fx, n, n, options.eps).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("powell_hybrid: initial jacobian - {}", e),
            }
        })?;

    let mut delta = tensor_norm(client, &x).map_err(|e| OptimizeError::NumericalError {
        message: format!("powell_hybrid: initial x norm - {}", e),
    })?;
    if delta == 0.0 {
        delta = 1.0;
    }
    delta = delta.min(100.0);

    let mut jac_age = 0; // How many iterations since full Jacobian recomputation
    let max_jac_age = n.max(5); // Recompute every n iterations at most

    for iter in 0..options.max_iter {
        // Check convergence
        let res_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
            message: format!("powell_hybrid: res norm - {}", e),
        })?;

        if res_norm < options.tol {
            return Ok(TensorRootResult {
                x,
                fun: fx,
                iterations: iter,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Compute Newton step: J * p_N = -F
        let neg_fx = client
            .mul_scalar(&fx, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: neg_fx - {}", e),
            })?;
        let neg_fx_col = neg_fx
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: reshape neg_fx - {}", e),
            })?;

        let newton_step = match LinearAlgebraAlgorithms::solve(client, &jac, &neg_fx_col) {
            Ok(p) => p.reshape(&[n]).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: reshape newton - {}", e),
            })?,
            Err(_) => {
                // Singular Jacobian: recompute and try again
                fx = f(&x).map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: f reeval - {}", e),
                })?;
                jac = finite_difference_jacobian(client, &f, &x, &fx, n, n, options.eps).map_err(
                    |e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: jac recompute - {}", e),
                    },
                )?;
                jac_age = 0;

                // Try again with fresh Jacobian, fall back to steepest descent
                match LinearAlgebraAlgorithms::solve(client, &jac, &neg_fx_col) {
                    Ok(p) => p.reshape(&[n]).map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: reshape newton retry - {}", e),
                    })?,
                    Err(_) => {
                        // Use steepest descent: -J^T F
                        let jt =
                            jac.transpose(0, 1)
                                .map_err(|e| OptimizeError::NumericalError {
                                    message: format!("powell_hybrid: transpose J - {}", e),
                                })?;
                        let fx_col =
                            fx.reshape(&[n, 1])
                                .map_err(|e| OptimizeError::NumericalError {
                                    message: format!("powell_hybrid: reshape fx sd - {}", e),
                                })?;
                        let jtf = client.matmul(&jt, &fx_col).map_err(|e| {
                            OptimizeError::NumericalError {
                                message: format!("powell_hybrid: J^T F - {}", e),
                            }
                        })?;
                        let sd = client.mul_scalar(&jtf, -1.0).map_err(|e| {
                            OptimizeError::NumericalError {
                                message: format!("powell_hybrid: neg J^T F - {}", e),
                            }
                        })?;
                        sd.reshape(&[n])
                            .map_err(|e| OptimizeError::NumericalError {
                                message: format!("powell_hybrid: reshape sd - {}", e),
                            })?
                    }
                }
            }
        };

        // Compute Newton step norm
        let newton_norm =
            tensor_norm(client, &newton_step).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: newton norm - {}", e),
            })?;

        // Dogleg step: choose step based on trust region
        let step = if newton_norm <= delta {
            // Newton step fits inside trust region
            newton_step
        } else {
            // Compute steepest descent direction: -J^T F
            let jt = jac
                .transpose(0, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: transpose J dogleg - {}", e),
                })?;
            let fx_col = fx
                .reshape(&[n, 1])
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: reshape fx dogleg - {}", e),
                })?;
            let jtf = client
                .matmul(&jt, &fx_col)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: J^T F dogleg - {}", e),
                })?;
            let sd = client
                .mul_scalar(
                    &jtf.reshape(&[n])
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("powell_hybrid: reshape jtf - {}", e),
                        })?,
                    -1.0,
                )
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: neg jtf - {}", e),
                })?;

            let sd_norm = tensor_norm(client, &sd).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: sd norm - {}", e),
            })?;

            if sd_norm == 0.0 {
                // Zero gradient: scale Newton step to trust region
                client
                    .mul_scalar(&newton_step, delta / newton_norm)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: scale newton - {}", e),
                    })?
            } else {
                // Cauchy point: optimal step along steepest descent
                let sd_dot_sd =
                    tensor_dot(client, &sd, &sd).map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: sd.sd - {}", e),
                    })?;

                // J @ sd
                let sd_col = sd
                    .reshape(&[n, 1])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: reshape sd col - {}", e),
                    })?;
                let j_sd =
                    client
                        .matmul(&jac, &sd_col)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("powell_hybrid: J @ sd - {}", e),
                        })?;
                let j_sd_flat = j_sd
                    .reshape(&[n])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: reshape j_sd - {}", e),
                    })?;
                let jsd_dot_jsd = tensor_dot(client, &j_sd_flat, &j_sd_flat).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("powell_hybrid: jsd.jsd - {}", e),
                    }
                })?;

                let cauchy_alpha = if jsd_dot_jsd > 0.0 {
                    sd_dot_sd / jsd_dot_jsd
                } else {
                    delta / sd_norm
                };

                let cauchy_point = client.mul_scalar(&sd, cauchy_alpha).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("powell_hybrid: cauchy point - {}", e),
                    }
                })?;
                let cauchy_norm = tensor_norm(client, &cauchy_point).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("powell_hybrid: cauchy norm - {}", e),
                    }
                })?;

                if cauchy_norm >= delta {
                    // Steepest descent clipped to trust region
                    client.mul_scalar(&sd, delta / sd_norm).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("powell_hybrid: sd clipped - {}", e),
                        }
                    })?
                } else {
                    // Dogleg: interpolate between Cauchy point and Newton step
                    // Find tau such that ||cauchy + tau*(newton - cauchy)|| = delta
                    let diff = client.sub(&newton_step, &cauchy_point).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("powell_hybrid: newton-cauchy - {}", e),
                        }
                    })?;

                    let a = tensor_dot(client, &diff, &diff).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("powell_hybrid: diff.diff - {}", e),
                        }
                    })?;
                    let b = 2.0
                        * tensor_dot(client, &cauchy_point, &diff).map_err(|e| {
                            OptimizeError::NumericalError {
                                message: format!("powell_hybrid: cauchy.diff - {}", e),
                            }
                        })?;
                    let c_val = cauchy_norm * cauchy_norm - delta * delta;

                    let discriminant = b * b - 4.0 * a * c_val;
                    let tau = if discriminant > 0.0 && a > 0.0 {
                        ((-b + discriminant.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let tau_diff = client.mul_scalar(&diff, tau).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("powell_hybrid: tau*diff - {}", e),
                        }
                    })?;
                    client.add(&cauchy_point, &tau_diff).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("powell_hybrid: dogleg step - {}", e),
                        }
                    })?
                }
            }
        };

        // Evaluate at trial point
        let x_new = client
            .add(&x, &step)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: x + step - {}", e),
            })?;
        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("powell_hybrid: f(x_new) - {}", e),
        })?;

        let new_res_norm =
            tensor_norm(client, &fx_new).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: new res norm - {}", e),
            })?;

        // Compute predicted reduction: ||F||^2 - ||F + J*step||^2
        let step_col = step
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: reshape step - {}", e),
            })?;
        let j_step = client
            .matmul(&jac, &step_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: J*step - {}", e),
            })?;
        let j_step_flat = j_step
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: reshape j_step - {}", e),
            })?;
        let predicted_fx =
            client
                .add(&fx, &j_step_flat)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: F + J*step - {}", e),
                })?;
        let predicted_norm =
            tensor_norm(client, &predicted_fx).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell_hybrid: predicted norm - {}", e),
            })?;

        let actual_reduction = res_norm * res_norm - new_res_norm * new_res_norm;
        let predicted_reduction = res_norm * res_norm - predicted_norm * predicted_norm;

        // Compute ratio
        let rho = if predicted_reduction.abs() < 1e-30 {
            if actual_reduction.abs() < 1e-30 {
                1.0
            } else {
                0.0
            }
        } else {
            actual_reduction / predicted_reduction
        };

        // Update trust region radius
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 {
            let step_norm =
                tensor_norm(client, &step).map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: step norm - {}", e),
                })?;
            if (step_norm - delta).abs() / delta < 0.1 {
                delta *= 2.0;
            }
        }

        // Accept or reject step
        if rho > 0.1 {
            // Accept: apply Broyden rank-1 update to Jacobian
            let df = client
                .sub(&fx_new, &fx)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: df - {}", e),
                })?;

            // Broyden update: J_new = J + (df - J*dx) * dx^T / (dx^T * dx)
            let dx_dot_dx =
                tensor_dot(client, &step, &step).map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: dx.dx - {}", e),
                })?;

            if dx_dot_dx > 1e-30 {
                // u = df - J*dx
                let u =
                    client
                        .sub(&df, &j_step_flat)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("powell_hybrid: u = df - J*dx - {}", e),
                        })?;

                // u * dx^T / (dx^T * dx) -> rank-1 update
                let u_col = u
                    .reshape(&[n, 1])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: reshape u - {}", e),
                    })?;
                let dx_row = step
                    .reshape(&[1, n])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: reshape dx row - {}", e),
                    })?;
                let outer =
                    client
                        .matmul(&u_col, &dx_row)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("powell_hybrid: outer product - {}", e),
                        })?;
                let update = client.mul_scalar(&outer, 1.0 / dx_dot_dx).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("powell_hybrid: scale update - {}", e),
                    }
                })?;
                jac = client
                    .add(&jac, &update)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: J update - {}", e),
                    })?;
                jac_age += 1;
            }

            x = x_new;
            fx = fx_new;

            // Check step convergence
            let step_norm =
                tensor_norm(client, &step).map_err(|e| OptimizeError::NumericalError {
                    message: format!("powell_hybrid: step norm check - {}", e),
                })?;
            if step_norm < options.x_tol {
                return Ok(TensorRootResult {
                    x,
                    fun: fx,
                    iterations: iter + 1,
                    residual_norm: new_res_norm,
                    converged: new_res_norm < options.tol,
                });
            }

            // Periodically recompute full Jacobian
            if jac_age >= max_jac_age {
                jac = finite_difference_jacobian(client, &f, &x, &fx, n, n, options.eps).map_err(
                    |e| OptimizeError::NumericalError {
                        message: format!("powell_hybrid: jac refresh - {}", e),
                    },
                )?;
                jac_age = 0;
            }
        }

        // Check if trust region collapsed
        if delta < 1e-15 {
            break;
        }
    }

    // Did not converge
    let final_norm = tensor_norm(client, &fx).map_err(|e| OptimizeError::NumericalError {
        message: format!("powell_hybrid: final norm - {}", e),
    })?;

    Ok(TensorRootResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        residual_norm: final_norm,
        converged: false,
    })
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
    fn test_powell_hybrid_linear() {
        let (device, client) = setup();

        // F(x) = [x1 + x2 - 3, 2*x1 - x2], solution: x = [1, 2]
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();

        let result = powell_hybrid_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let residuals = vec![data[0] + data[1] - 3.0, 2.0 * data[0] - data[1]];
                Tensor::from_slice(&residuals, &[2], x.device())
            },
            &x0,
            &RootOptions::default(),
        )
        .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 1e-6);
        assert!((sol[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_powell_hybrid_quadratic() {
        let (device, client) = setup();

        // x^2 + y^2 = 1, x = y => x = y = 1/sqrt(2)
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[2], &device).unwrap();

        let result = powell_hybrid_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let residuals = vec![
                    data[0] * data[0] + data[1] * data[1] - 1.0,
                    data[0] - data[1],
                ];
                Tensor::from_slice(&residuals, &[2], x.device())
            },
            &x0,
            &RootOptions::default(),
        )
        .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sol[0] - expected).abs() < 1e-6);
        assert!((sol[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_powell_hybrid_broyden_tridiagonal() {
        let (device, client) = setup();

        // Broyden tridiagonal: (3-2*x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1 = 0
        let n = 5;
        let x0 = Tensor::<CpuRuntime>::from_slice(&[-1.0; 5], &[n], &device).unwrap();

        let result = powell_hybrid_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let n = data.len();
                let mut residuals = vec![0.0; n];
                for i in 0..n {
                    let x_prev = if i > 0 { data[i - 1] } else { 0.0 };
                    let x_next = if i < n - 1 { data[i + 1] } else { 0.0 };
                    residuals[i] = (3.0 - 2.0 * data[i]) * data[i] - x_prev - 2.0 * x_next + 1.0;
                }
                Tensor::from_slice(&residuals, x.shape(), x.device())
            },
            &x0,
            &RootOptions {
                max_iter: 200,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            result.converged,
            "Powell hybrid did not converge on Broyden tridiagonal"
        );
        assert!(result.residual_norm < 1e-6);
    }
}
