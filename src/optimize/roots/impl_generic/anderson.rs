//! Anderson mixing implementation for accelerating fixed-point iterations.
//!
//! Anderson mixing (also known as Anderson acceleration or DIIS) accelerates
//! convergence of fixed-point iterations x_{k+1} = g(x_k) by combining
//! the m most recent iterates using least-squares optimal coefficients.
//!
//! # Algorithm
//!
//! Given iterates x_0, ..., x_k and residuals f_i = g(x_i) - x_i:
//! 1. Build matrix F = [f_{k-m+1} - f_k, ..., f_{k-1} - f_k] (columns are residual differences)
//! 2. Solve min ||F @ theta + f_k||^2 via least squares
//! 3. x_{k+1} = (1 - alpha) * sum(theta_i * x_i) + alpha * sum(theta_i * g(x_i))
//!    where coefficients sum to 1
use crate::DType;

use std::collections::VecDeque;

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::utils::tensor_norm;
use crate::optimize::roots::traits::anderson::{AndersonConvergence, AndersonOptions};

use super::TensorRootResult;

#[derive(Clone, Copy)]
enum ConvergenceStage {
    Iteration(usize),
    Limit(usize),
}

/// Anderson mixing with the built-in residual norm test.
pub fn anderson_impl<R, C, G>(
    client: &C,
    g: G,
    x0: &Tensor<R>,
    options: &AndersonOptions,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    G: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    anderson_engine(client, g, x0, options, |_, _, fx, stage| {
        let context = match stage {
            ConvergenceStage::Iteration(_) => "norm",
            ConvergenceStage::Limit(_) => "final norm",
        };
        let residual_norm = tensor_norm(client, fx).map_err(|e| OptimizeError::NumericalError {
            message: format!("anderson: {context} - {e}"),
        })?;
        Ok(AndersonConvergence {
            residual_norm,
            converged: residual_norm < options.base.tol,
        })
    })
}

/// Anderson mixing with caller-defined stopping conditions.
pub fn anderson_with_convergence_impl<R, C, G, K>(
    client: &C,
    g: G,
    x0: &Tensor<R>,
    options: &AndersonOptions,
    convergence: K,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    G: Fn(&Tensor<R>) -> Result<Tensor<R>>,
    K: Fn(&Tensor<R>, &Tensor<R>, &Tensor<R>) -> Result<AndersonConvergence>,
{
    anderson_engine(client, g, x0, options, |x, gx, fx, stage| {
        let decision = convergence(x, gx, fx).map_err(|e| OptimizeError::NumericalError {
            message: format!(
                "anderson: {} convergence evaluation - {e}",
                stage_description(stage)
            ),
        })?;
        check_custom_convergence(decision, stage)
    })
}

fn anderson_engine<R, C, G, K>(
    client: &C,
    g: G,
    x0: &Tensor<R>,
    options: &AndersonOptions,
    convergence: K,
) -> OptimizeResult<TensorRootResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    G: Fn(&Tensor<R>) -> Result<Tensor<R>>,
    K: Fn(
        &Tensor<R>,
        &Tensor<R>,
        &Tensor<R>,
        ConvergenceStage,
    ) -> OptimizeResult<AndersonConvergence>,
{
    let [n] = x0.shape() else {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "anderson: initial shape={:?}. Supply a nonempty one-dimensional guess",
                x0.shape()
            ),
        });
    };
    if *n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "anderson: initial shape=[0]. Supply a nonempty one-dimensional guess".into(),
        });
    }

    let m = options.m;
    let alpha = options.alpha;
    let opts = &options.base;

    let mut x = x0.clone();

    // Evaluate g(x0)
    let mut gx = g(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("anderson: initial g eval - {}", e),
    })?;
    check_map_shape(&gx, x0.shape())?;

    // Residual: f = g(x) - x
    let mut fx = client
        .sub(&gx, &x)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("anderson: initial residual - {}", e),
        })?;

    // History storage: store previous x values and residuals (VecDeque for O(1) pop_front)
    let mut x_history: VecDeque<Tensor<R>> = VecDeque::with_capacity(m + 2);
    let mut f_history: VecDeque<Tensor<R>> = VecDeque::with_capacity(m + 2);

    x_history.push_back(x.clone());
    f_history.push_back(fx.clone());

    for iter in 0..opts.max_iter {
        let decision = convergence(&x, &gx, &fx, ConvergenceStage::Iteration(iter))?;

        if decision.converged {
            return Ok(TensorRootResult {
                x,
                fun: fx,
                iterations: iter,
                residual_norm: decision.residual_norm,
                converged: true,
            });
        }

        let mk = f_history.len(); // Current history size

        if m == 0 && alpha == 1.0 {
            // Reconstructing gx as x + (gx - x) can lose its original bits.
            x = gx;
        } else if mk < 2 {
            // Not enough history yet: simple fixed-point step
            // x_new = (1 - alpha) * x + alpha * g(x) = x + alpha * f
            let step =
                client
                    .mul_scalar(&fx, alpha)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: simple step - {}", e),
                    })?;
            x = client
                .add(&x, &step)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("anderson: x update simple - {}", e),
                })?;
        } else {
            // Anderson mixing with least-squares
            let num_cols = mk - 1; // Number of residual differences

            // Build the difference matrix: columns are (f_i - f_{mk-1}) for i = 0..mk-2
            let f_last = &f_history[mk - 1];
            let mut diff_cols: Vec<Tensor<R>> = Vec::with_capacity(num_cols);

            #[allow(clippy::needless_range_loop)]
            for i in 0..num_cols {
                let diff = client.sub(&f_history[i], f_last).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("anderson: f diff {} - {}", i, e),
                    }
                })?;
                let col = diff
                    .unsqueeze(1)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: unsqueeze diff - {}", e),
                    })?;
                diff_cols.push(col);
            }

            let refs: Vec<&Tensor<R>> = diff_cols.iter().collect();
            let diff_matrix = client
                .cat(&refs, 1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("anderson: cat diff matrix - {}", e),
                })?; // [n, num_cols]

            // Solve: min ||diff_matrix @ theta - (-f_last)||^2
            // i.e., diff_matrix @ theta ≈ -f_last
            let neg_f_last =
                client
                    .mul_scalar(f_last, -1.0)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: neg f_last - {}", e),
                    })?;
            let rhs = neg_f_last
                .unsqueeze(1)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("anderson: unsqueeze rhs - {}", e),
                })?; // [n, 1]

            // Least squares solve
            let theta =
                LinearAlgebraAlgorithms::lstsq(client, &diff_matrix, &rhs).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("anderson: lstsq - {}", e),
                    }
                })?; // [num_cols, 1]

            let theta_flat =
                theta
                    .reshape(&[num_cols])
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: reshape theta - {}", e),
                    })?;

            // Compute mixing coefficients: gamma_i = theta_i, gamma_last = 1 - sum(theta)
            // theta_sum is a single scalar extracted on device
            let theta_sum_tensor = client.sum(&theta_flat, &[0], false).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("anderson: theta sum - {}", e),
                }
            })?;
            let theta_sum: f64 =
                theta_sum_tensor
                    .item::<f64>()
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: theta sum scalar - {}", e),
                    })?;

            // x_new = sum_i gamma_i * (x_i + alpha * f_i)
            // where gamma = [theta_0, ..., theta_{mk-2}, 1 - sum(theta)]
            let gamma_last = 1.0 - theta_sum;

            // Start with the contribution from the last iterate
            let x_plus_af_last = {
                let af = client.mul_scalar(f_last, alpha).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("anderson: alpha*f_last - {}", e),
                    }
                })?;
                client
                    .add(&x_history[mk - 1], &af)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("anderson: x+af last - {}", e),
                    })?
            };
            let mut x_new = client
                .mul_scalar(&x_plus_af_last, gamma_last)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("anderson: gamma_last * term - {}", e),
                })?;

            // Add contributions from earlier iterates
            // Extract each theta_i as a scalar (m is small, typically 5)
            for i in 0..num_cols {
                let theta_i_tensor =
                    theta_flat
                        .narrow(0, i, 1)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("anderson: narrow theta {} - {}", i, e),
                        })?;
                let theta_i: f64 =
                    theta_i_tensor
                        .item::<f64>()
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("anderson: theta_{} scalar - {}", i, e),
                        })?;

                let x_plus_af = {
                    let af = client.mul_scalar(&f_history[i], alpha).map_err(|e| {
                        OptimizeError::NumericalError {
                            message: format!("anderson: alpha*f_{} - {}", i, e),
                        }
                    })?;
                    client
                        .add(&x_history[i], &af)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("anderson: x+af {} - {}", i, e),
                        })?
                };
                let contrib = client.mul_scalar(&x_plus_af, theta_i).map_err(|e| {
                    OptimizeError::NumericalError {
                        message: format!("anderson: theta*term {} - {}", i, e),
                    }
                })?;
                x_new =
                    client
                        .add(&x_new, &contrib)
                        .map_err(|e| OptimizeError::NumericalError {
                            message: format!("anderson: accumulate {} - {}", i, e),
                        })?;
            }

            x = x_new;
        }

        // Evaluate g at new x
        gx = g(&x).map_err(|e| OptimizeError::NumericalError {
            message: format!("anderson: g eval iter {} - {}", iter, e),
        })?;
        check_map_shape(&gx, x0.shape())?;
        fx = client
            .sub(&gx, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("anderson: residual iter {} - {}", iter, e),
            })?;

        // Update history (sliding window)
        x_history.push_back(x.clone());
        f_history.push_back(fx.clone());

        if x_history.len() > m + 1 {
            x_history.pop_front();
            f_history.pop_front();
        }
    }

    let final_decision = convergence(&x, &gx, &fx, ConvergenceStage::Limit(opts.max_iter))?;

    Ok(TensorRootResult {
        x,
        fun: fx,
        iterations: opts.max_iter,
        residual_norm: final_decision.residual_norm,
        converged: false,
    })
}

fn stage_description(stage: ConvergenceStage) -> String {
    match stage {
        ConvergenceStage::Iteration(iteration) => format!("iteration {iteration}"),
        ConvergenceStage::Limit(iterations) => format!("final check after {iterations} updates"),
    }
}

fn check_custom_convergence(
    decision: AndersonConvergence,
    stage: ConvergenceStage,
) -> OptimizeResult<AndersonConvergence> {
    if decision.residual_norm.is_nan()
        || decision.residual_norm < 0.0
        || (decision.converged && !decision.residual_norm.is_finite())
    {
        return Err(OptimizeError::NumericalError {
            message: format!(
                "anderson: {} diagnostic residual_norm={},converged={}. Return a nonnegative metric, finite when converged",
                stage_description(stage),
                decision.residual_norm,
                decision.converged
            ),
        });
    }
    Ok(decision)
}

fn check_map_shape<R: Runtime<DType = DType>>(
    value: &Tensor<R>,
    expected: &[usize],
) -> OptimizeResult<()> {
    if value.shape() != expected {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "anderson: map shape={:?}, initial shape={expected:?}. Return one value per initial coordinate",
                value.shape()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "anderson_tests.rs"]
mod tests;
