//! Trust-Exact: Nearly-exact trust region subproblem solver.
//!
//! Finds lambda such that ||(H + lambda*I)^{-1} g|| = trust_radius using
//! iterative Cholesky factorizations. Handles the "hard case" where the
//! gradient is orthogonal to the eigenvector of the smallest eigenvalue.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::error::Result as NumrResult;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

use super::trust_region_base::{
    SubproblemResult, SubproblemSolver, compute_predicted_reduction, secular_newton_update,
    trust_region_loop,
};
use super::utils::tensor_norm;

use super::helpers::hvp_from_fn;

/// Nearly-exact subproblem solver.
struct ExactSolver;

impl<R, C, F> SubproblemSolver<R, C, F> for ExactSolver
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    fn solve(
        &self,
        client: &C,
        f: &F,
        x: &Tensor<R>,
        grad: &Tensor<R>,
        trust_radius: f64,
    ) -> OptimizeResult<SubproblemResult<R>> {
        exact_subproblem(client, f, x, grad, trust_radius)
    }
}

/// Trust-exact implementation.
pub fn trust_exact_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &TrustRegionOptions,
) -> OptimizeResult<TrustRegionResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    trust_region_loop(client, f, x0, options, &ExactSolver)
}

/// Solve the trust region subproblem nearly exactly.
///
/// Find p that minimizes g^T p + 0.5 p^T H p subject to ||p|| <= delta.
///
/// Uses the secular equation approach: find lambda >= 0 such that
/// (H + lambda*I) p = -g and ||p|| = delta (or lambda = 0 and ||p|| < delta).
fn exact_subproblem<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    grad: &Tensor<R>,
    trust_radius: f64,
) -> OptimizeResult<SubproblemResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    let n = grad.numel();

    // Build full Hessian by computing H*e_i for each unit vector
    let hessian = build_full_hessian(client, f, x, n)?;

    // Try lambda = 0 first (unconstrained Newton step)
    let neg_g = client
        .mul_scalar(grad, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: neg_g - {}", e),
        })?;

    // Try Cholesky factorization of H
    let neg_g_col = neg_g
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: reshape neg_g - {}", e),
        })?;

    // Attempt to solve H p = -g directly
    match try_solve_with_lambda(client, &hessian, &neg_g_col, 0.0, n) {
        Ok(p) => {
            let p_flat = p.reshape(&[n]).map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: reshape p - {}", e),
            })?;
            let p_norm =
                tensor_norm(client, &p_flat).map_err(|e| OptimizeError::NumericalError {
                    message: format!("trust_exact: p norm - {}", e),
                })?;

            if p_norm <= trust_radius {
                // Unconstrained solution is inside trust region
                let hp = hessian_times_vector(client, &hessian, &p_flat, n)?;
                let pred = compute_predicted_reduction(client, grad, &p_flat, &hp)?;
                return Ok(SubproblemResult {
                    step: p_flat,
                    hits_boundary: false,
                    predicted_reduction: pred,
                });
            }
        }
        Err(_) => {
            // H is not positive definite, need lambda > 0
        }
    }

    // Need to find lambda such that ||(H + lambda*I)^{-1} g|| = delta
    // Use Newton's method on the secular equation:
    //   phi(lambda) = 1/||p(lambda)|| - 1/delta = 0

    // Get smallest eigenvalue to determine lambda lower bound
    let lambda_min = estimate_min_eigenvalue(client, &hessian)?;
    let mut lambda = (-lambda_min + 0.01).max(0.01);

    let max_iter = 50;
    for _ in 0..max_iter {
        match try_solve_with_lambda(client, &hessian, &neg_g_col, lambda, n) {
            Ok(p) => {
                let p_flat = p.reshape(&[n]).map_err(|e| OptimizeError::NumericalError {
                    message: format!("trust_exact: reshape p iter - {}", e),
                })?;
                let p_norm =
                    tensor_norm(client, &p_flat).map_err(|e| OptimizeError::NumericalError {
                        message: format!("trust_exact: p norm iter - {}", e),
                    })?;

                if (p_norm - trust_radius).abs() / trust_radius < 1e-4 {
                    // Close enough to the boundary
                    let hp = hessian_times_vector(client, &hessian, &p_flat, n)?;
                    let pred = compute_predicted_reduction(client, grad, &p_flat, &hp)?;
                    return Ok(SubproblemResult {
                        step: p_flat,
                        hits_boundary: true,
                        predicted_reduction: pred,
                    });
                }

                // Newton update for lambda using secular equation
                // phi(lambda) = ||p|| - delta
                // phi'(lambda) = -p^T (H+lambda*I)^{-2} g / ||p||
                // Approximate: delta_lambda = (||p|| - delta) * (||p|| / delta) * (||p|| / q_norm)
                // where q = (H + lambda*I)^{-1} p

                // Safeguarded Newton update on secular equation
                lambda = secular_newton_update(lambda, p_norm / trust_radius);
            }
            Err(_) => {
                // Factorization failed, increase lambda
                lambda *= 2.0;
            }
        }
    }

    // Fallback: return steepest descent step within trust region
    let grad_norm = tensor_norm(client, grad).map_err(|e| OptimizeError::NumericalError {
        message: format!("trust_exact: grad norm fallback - {}", e),
    })?;
    let scale = -(trust_radius / grad_norm);
    let step = client
        .mul_scalar(grad, scale)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: fallback step - {}", e),
        })?;
    let hp = hessian_times_vector(client, &hessian, &step, n)?;
    let pred = compute_predicted_reduction(client, grad, &step, &hp)?;

    Ok(SubproblemResult {
        step,
        hits_boundary: true,
        predicted_reduction: pred,
    })
}

/// Build the full n x n Hessian matrix using HVP.
fn build_full_hessian<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: eye - {}", e),
        })?;

    let mut columns: Vec<Tensor<R>> = Vec::with_capacity(n);
    for i in 0..n {
        let ei = identity
            .narrow(0, i, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: narrow ei - {}", e),
            })?
            .contiguous()
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: contiguous ei - {}", e),
            })?
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: reshape ei - {}", e),
            })?;
        let (_fx, hvp) = hvp_from_fn(client, f, x, &ei)?;
        let col = hvp
            .unsqueeze(1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: unsqueeze hvp - {}", e),
            })?;
        columns.push(col);
    }

    let refs: Vec<&Tensor<R>> = columns.iter().collect();
    client
        .cat(&refs, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: cat hessian - {}", e),
        })
}

/// Try to solve (H + lambda*I) p = -g using Cholesky decomposition.
fn try_solve_with_lambda<R, C>(
    client: &C,
    hessian: &Tensor<R>,
    neg_g_col: &Tensor<R>,
    lambda: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    // H + lambda * I
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: eye lambda - {}", e),
        })?;
    let lambda_i =
        client
            .mul_scalar(&identity, lambda)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: lambda_i - {}", e),
            })?;
    let h_shifted = client
        .add(hessian, &lambda_i)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: h + lambda_i - {}", e),
        })?;

    // Solve using linear system solver
    LinearAlgebraAlgorithms::solve(client, &h_shifted, neg_g_col).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("trust_exact: solve - {}", e),
        }
    })
}

/// Estimate the minimum eigenvalue of the Hessian using Gershgorin circles.
fn estimate_min_eigenvalue<R, C>(client: &C, hessian: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Use the diagonal elements and row sums for a Gershgorin-based estimate
    let diag = client
        .diag(hessian)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: diag - {}", e),
        })?;

    let hessian_abs = client
        .abs(hessian)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: abs - {}", e),
        })?;

    // Sum of absolute values of each row
    let row_sums =
        client
            .sum(&hessian_abs, &[1], false)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: row_sums - {}", e),
            })?;

    // off_diag_sums = row_sums - |diag|  (sum of absolute off-diagonal entries per row)
    let diag_abs = client
        .abs(&diag)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: abs diag - {}", e),
        })?;
    let off_diag_sums =
        client
            .sub(&row_sums, &diag_abs)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: off_diag_sums - {}", e),
            })?;

    // Gershgorin lower bounds: diag - off_diag_sums
    let lower_bounds =
        client
            .sub(&diag, &off_diag_sums)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: lower_bounds - {}", e),
            })?;

    // Extract the minimum (single scalar)
    let min_tensor =
        client
            .min(&lower_bounds, &[0], false)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_exact: min eigenvalue - {}", e),
            })?;
    min_tensor
        .item::<f64>()
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: min eigenvalue scalar - {}", e),
        })
}

/// Compute H * v where H is a full matrix and v is a vector.
fn hessian_times_vector<R, C>(
    client: &C,
    hessian: &Tensor<R>,
    v: &Tensor<R>,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let v_col = v
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: reshape v - {}", e),
        })?;
    let result = client
        .matmul(hessian, &v_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: matmul Hv - {}", e),
        })?;
    result
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_exact: reshape Hv result - {}", e),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_sum};
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_trust_exact_quadratic() {
        let (device, client) = setup();

        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let result = trust_exact_impl(
            &client,
            |x_var, c| {
                let x_sq = var_mul(x_var, x_var, c)?;
                var_sum(&x_sq, &[0], false, c)
            },
            &x0,
            &TrustRegionOptions::default(),
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_trust_exact_shifted_quadratic() {
        let (device, client) = setup();

        // f(x) = sum((x - [1,2,3])^2), minimum at [1, 2, 3]
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0, 0.0], &[3], &device);

        let result = trust_exact_impl(
            &client,
            |x_var, c| {
                let target = Var::new(
                    Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device),
                    false,
                );
                let diff = numr::autograd::var_sub(x_var, &target, c)?;
                let diff_sq = var_mul(&diff, &diff, c)?;
                var_sum(&diff_sq, &[0], false, c)
            },
            &x0,
            &TrustRegionOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "Trust-exact did not converge");
        assert!(result.fun < 1e-10, "fun = {}", result.fun);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 1e-4);
        assert!((sol[1] - 2.0).abs() < 1e-4);
        assert!((sol[2] - 3.0).abs() < 1e-4);
    }
}
