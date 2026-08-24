//! Trust-Krylov: GLTR (Generalized Lanczos Trust Region) subproblem solver.
//!
//! Uses Lanczos iteration to build a tridiagonal approximation T of the Hessian H,
//! then solves the trust region subproblem in the reduced Krylov space.
//! The solution is mapped back to the full space using the Lanczos vectors.
use crate::DType;

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

use super::trust_region_base::{
    SubproblemResult, SubproblemSolver, compute_hvp_for_subproblem, compute_predicted_reduction,
    secular_newton_update, trust_region_loop,
};
use super::utils::{tensor_dot, tensor_norm};

/// GLTR Krylov subproblem solver.
struct KrylovSolver;

impl<R, C, F> SubproblemSolver<R, C, F> for KrylovSolver
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
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
        gltr_subproblem(client, f, x, grad, trust_radius)
    }
}

/// Trust-Krylov implementation.
pub fn trust_krylov_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &TrustRegionOptions,
) -> OptimizeResult<TrustRegionResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    trust_region_loop(client, f, x0, options, &KrylovSolver)
}

/// GLTR subproblem solver using Lanczos iteration.
///
/// 1. Build Lanczos tridiagonal matrix T and orthonormal basis Q
/// 2. Solve the small trust region subproblem: min e1^T ||g|| y + 0.5 y^T T y  s.t. ||y|| <= delta
/// 3. Map back: p = Q @ y
fn gltr_subproblem<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    grad: &Tensor<R>,
    trust_radius: f64,
) -> OptimizeResult<SubproblemResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    let n = grad.numel();
    let max_lanczos = n.min(100); // Lanczos iterations

    let grad_norm = tensor_norm(client, grad).map_err(|e| OptimizeError::NumericalError {
        message: format!("gltr: grad norm - {}", e),
    })?;

    if grad_norm < 1e-30 {
        // Zero gradient: return zero step
        let step = Tensor::<R>::zeros(grad.shape(), grad.dtype(), grad.device())?;
        return Ok(SubproblemResult {
            step,
            hits_boundary: false,
            predicted_reduction: 0.0,
        });
    }

    // Lanczos iteration
    // q1 = -g / ||g||
    let q1 =
        client
            .mul_scalar(grad, -1.0 / grad_norm)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("gltr: q1 - {}", e),
            })?;

    let mut lanczos_vecs: Vec<Tensor<R>> = Vec::with_capacity(max_lanczos);
    let mut alphas: Vec<f64> = Vec::with_capacity(max_lanczos);
    let mut betas: Vec<f64> = Vec::with_capacity(max_lanczos);

    lanczos_vecs.push(q1);
    let mut beta_prev = 0.0;
    let mut q_prev = Tensor::<R>::zeros(grad.shape(), grad.dtype(), grad.device())?;

    for k in 0..max_lanczos {
        let q_k = &lanczos_vecs[k];

        // w = H * q_k
        let w = compute_hvp_for_subproblem(client, f, x, q_k)?;

        // alpha_k = q_k^T * w
        let alpha_k = tensor_dot(client, q_k, &w).map_err(|e| OptimizeError::NumericalError {
            message: format!("gltr: dot - {}", e),
        })?;
        alphas.push(alpha_k);

        // w = w - alpha_k * q_k - beta_{k-1} * q_{k-1}
        let alpha_qk =
            client
                .mul_scalar(q_k, alpha_k)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("gltr: alpha_qk - {}", e),
                })?;
        let mut w_new = client
            .sub(&w, &alpha_qk)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("gltr: w - alpha_qk - {}", e),
            })?;

        if k > 0 {
            let beta_qprev = client.mul_scalar(&q_prev, beta_prev).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("gltr: beta_qprev - {}", e),
                }
            })?;
            w_new = client
                .sub(&w_new, &beta_qprev)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("gltr: w - beta_qprev - {}", e),
                })?;
        }

        // Reorthogonalize against all previous Lanczos vectors (full reorthogonalization)
        for q_j in &lanczos_vecs {
            let proj =
                tensor_dot(client, &w_new, q_j).map_err(|e| OptimizeError::NumericalError {
                    message: format!("gltr: reortho dot - {}", e),
                })?;
            let proj_qj =
                client
                    .mul_scalar(q_j, proj)
                    .map_err(|e| OptimizeError::NumericalError {
                        message: format!("gltr: reortho - {}", e),
                    })?;
            w_new = client
                .sub(&w_new, &proj_qj)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("gltr: reortho sub - {}", e),
                })?;
        }

        let beta_k = tensor_norm(client, &w_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("gltr: beta_k norm - {}", e),
        })?;

        if beta_k < 1e-12 {
            // Invariant subspace found, break
            break;
        }

        betas.push(beta_k);
        q_prev = q_k.clone();
        beta_prev = beta_k;

        let q_next =
            client
                .mul_scalar(&w_new, 1.0 / beta_k)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("gltr: q_next - {}", e),
                })?;
        lanczos_vecs.push(q_next);
    }

    let k = alphas.len(); // actual Lanczos dimension

    // Solve small trust region subproblem in Krylov space
    // min ||g|| * e1^T y + 0.5 y^T T y  s.t. ||y|| <= delta
    // where T is the k x k tridiagonal matrix with alphas on diagonal, betas on off-diagonals
    // and the first element of the gradient in Krylov space is ||g||

    let y = solve_tridiagonal_trust_region(&alphas, &betas, grad_norm, trust_radius, k)?;

    // Map back to full space: p = Q @ y = sum_i y_i * q_i
    let mut step = Tensor::<R>::zeros(grad.shape(), grad.dtype(), grad.device())?;
    for i in 0..y.len() {
        let yi_qi = client.mul_scalar(&lanczos_vecs[i], y[i]).map_err(|e| {
            OptimizeError::NumericalError {
                message: format!("gltr: yi_qi - {}", e),
            }
        })?;
        step = client
            .add(&step, &yi_qi)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("gltr: step accumulate - {}", e),
            })?;
    }

    // Compute predicted reduction
    let hp = compute_hvp_for_subproblem(client, f, x, &step)?;
    let pred = compute_predicted_reduction(client, grad, &step, &hp)?;

    let step_norm = tensor_norm(client, &step).map_err(|e| OptimizeError::NumericalError {
        message: format!("gltr: step norm - {}", e),
    })?;

    Ok(SubproblemResult {
        step,
        hits_boundary: (step_norm - trust_radius).abs() / trust_radius < 0.05,
        predicted_reduction: pred,
    })
}

/// Solve the small tridiagonal trust region subproblem.
///
/// min ||g|| * e1^T y + 0.5 y^T T y  subject to ||y|| <= delta
///
/// This is solved exactly using the secular equation approach on the small k x k system.
fn solve_tridiagonal_trust_region(
    alphas: &[f64],
    betas: &[f64],
    grad_norm: f64,
    trust_radius: f64,
    k: usize,
) -> OptimizeResult<Vec<f64>> {
    if k == 0 {
        return Ok(vec![]);
    }

    // Unconstrained solution: T y = -(Q^T g) = ||g|| * e1
    // since q1 = -g/||g||, so Q^T g = [-||g||, 0, ...], and -(Q^T g) = [||g||, 0, ...]
    let mut rhs = vec![0.0; k];
    rhs[0] = grad_norm;

    // Solve tridiagonal system T y = rhs using Thomas algorithm
    if let Some(y) = solve_tridiagonal(alphas, betas, &rhs) {
        let y_norm: f64 = y.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
        if y_norm <= trust_radius {
            return Ok(y);
        }
    }

    // Need constrained solution: find lambda such that ||y(lambda)|| = delta
    // where (T + lambda*I) y = -||g|| e1
    let mut lambda = 0.01;
    let max_iter = 100;

    for _ in 0..max_iter {
        // Shifted diagonal
        let shifted_alphas: Vec<f64> = alphas.iter().map(|a| a + lambda).collect();

        if let Some(y) = solve_tridiagonal(&shifted_alphas, betas, &rhs) {
            let y_norm: f64 = y.iter().map(|yi| yi * yi).sum::<f64>().sqrt();

            if (y_norm - trust_radius).abs() / trust_radius < 1e-3 {
                return Ok(y);
            }

            // Safeguarded Newton update on secular equation
            lambda = secular_newton_update(lambda, y_norm / trust_radius);
        } else {
            lambda *= 2.0;
        }
    }

    // Fallback: steepest descent in Krylov space
    // q1 = -g/||g||, so y[0] > 0 means step in -g direction (descent)
    let mut y = vec![0.0; k];
    y[0] = trust_radius;
    Ok(y)
}

/// Solve tridiagonal system using Thomas algorithm.
/// Returns None if the system is singular.
fn solve_tridiagonal(diag: &[f64], off_diag: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Some(vec![]);
    }
    if n == 1 {
        if diag[0].abs() < 1e-15 {
            return None;
        }
        return Some(vec![rhs[0] / diag[0]]);
    }

    // Forward elimination
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = if off_diag.is_empty() {
        0.0
    } else {
        off_diag[0] / diag[0]
    };
    d_prime[0] = rhs[0] / diag[0];

    if diag[0].abs() < 1e-15 {
        return None;
    }

    for i in 1..n {
        let beta_i = if i - 1 < off_diag.len() {
            off_diag[i - 1]
        } else {
            0.0
        };
        let denom = diag[i] - beta_i * c_prime[i - 1];
        if denom.abs() < 1e-15 {
            return None;
        }
        c_prime[i] = if i < off_diag.len() {
            off_diag[i] / denom
        } else {
            0.0
        };
        d_prime[i] = (rhs[i] - beta_i * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Some(x)
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
    fn test_trust_krylov_quadratic() {
        let (device, client) = setup();

        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let result = trust_krylov_impl(
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
    fn test_trust_krylov_sphere_10d() {
        let (device, client) = setup();

        let initial: Vec<f64> = (0..10).map(|i| (i as f64) - 5.0).collect();
        let x0 = Tensor::<CpuRuntime>::from_slice(&initial, &[10], &device).unwrap();

        let result = trust_krylov_impl(
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
    fn test_tridiagonal_solve() {
        // Simple 3x3 tridiagonal: [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        let diag = vec![2.0, 3.0, 2.0];
        let off = vec![1.0, 1.0];
        let rhs = vec![1.0, 2.0, 3.0];
        let y = solve_tridiagonal(&diag, &off, &rhs).unwrap();
        // Verify: T @ y ≈ rhs
        assert!((2.0 * y[0] + 1.0 * y[1] - 1.0).abs() < 1e-10);
        assert!((1.0 * y[0] + 3.0 * y[1] + 1.0 * y[2] - 2.0).abs() < 1e-10);
        assert!((1.0 * y[1] + 2.0 * y[2] - 3.0).abs() < 1e-10);
    }
}
