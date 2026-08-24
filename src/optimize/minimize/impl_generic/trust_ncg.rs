//! Trust-NCG: Trust region Newton-CG with Steihaug-Toint CG subproblem.
//!
//! The Steihaug-Toint CG algorithm solves the trust region subproblem:
//!   min g^T p + 0.5 p^T H p   s.t. ||p|| <= delta
//!
//! by running conjugate gradient on Hp = -g, truncating when:
//! 1. Negative curvature is encountered (d^T H d <= 0)
//! 2. The iterates hit the trust region boundary (||p|| >= delta)
//! 3. CG converges (||r|| < tol)
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
    trust_region_loop,
};
use super::utils::{tensor_dot, tensor_norm};

/// Steihaug-Toint CG subproblem solver.
struct SteihaugCG;

impl<R, C, F> SubproblemSolver<R, C, F> for SteihaugCG
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
        steihaug_toint_cg(client, f, x, grad, trust_radius)
    }
}

/// Trust-NCG implementation.
///
/// Combines the trust region outer loop with Steihaug-Toint CG subproblem solver.
pub fn trust_ncg_impl<R, C, F>(
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
    trust_region_loop(client, f, x0, options, &SteihaugCG)
}

/// Steihaug-Toint truncated CG for the trust region subproblem.
///
/// Solves: min g^T p + 0.5 p^T H p  subject to ||p|| <= delta
///
/// The key insight: if we encounter negative curvature or hit the boundary,
/// we move to the boundary along the current direction and stop.
fn steihaug_toint_cg<R, C, F>(
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
    let max_cg_iter = n.min(200);
    let device = grad.device();
    let dtype = grad.dtype();

    let grad_norm = tensor_norm(client, grad).map_err(|e| OptimizeError::NumericalError {
        message: format!("steihaug: grad norm - {}", e),
    })?;

    // Tolerance for CG convergence: Eisenstat-Walker forcing sequence
    // tol = min(0.5, sqrt(grad_norm)) * grad_norm
    let tol = 0.5_f64.min(grad_norm.sqrt()) * grad_norm;

    // Initialize: p = 0, r = -g, d = -g
    let mut p = Tensor::<R>::zeros(grad.shape(), dtype, device)?;
    let neg_g = client
        .mul_scalar(grad, -1.0)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("steihaug: neg_g - {}", e),
        })?;
    let mut r = neg_g.clone();
    let mut d = neg_g;

    let mut r_dot_r = tensor_dot(client, &r, &r).map_err(|e| OptimizeError::NumericalError {
        message: format!("steihaug: dot - {}", e),
    })?;

    for _ in 0..max_cg_iter {
        // Check CG convergence
        if r_dot_r.sqrt() < tol {
            // Converged: compute predicted reduction
            let hp = compute_hvp_for_subproblem(client, f, x, &p)?;
            let pred = compute_predicted_reduction(client, grad, &p, &hp)?;
            return Ok(SubproblemResult {
                step: p,
                hits_boundary: false,
                predicted_reduction: pred,
            });
        }

        // Compute H·d
        let hd = compute_hvp_for_subproblem(client, f, x, &d)?;
        let d_dot_hd = tensor_dot(client, &d, &hd).map_err(|e| OptimizeError::NumericalError {
            message: format!("steihaug: dot - {}", e),
        })?;

        // Check for negative curvature
        if d_dot_hd <= 0.0 {
            // Move to trust region boundary along d
            let p_boundary = move_to_boundary(client, &p, &d, trust_radius)?;
            let hp = compute_hvp_for_subproblem(client, f, x, &p_boundary)?;
            let pred = compute_predicted_reduction(client, grad, &p_boundary, &hp)?;
            return Ok(SubproblemResult {
                step: p_boundary,
                hits_boundary: true,
                predicted_reduction: pred,
            });
        }

        // CG step size
        let alpha = r_dot_r / d_dot_hd;

        // Tentative new p
        let alpha_d = client
            .mul_scalar(&d, alpha)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: alpha_d - {}", e),
            })?;
        let p_new = client
            .add(&p, &alpha_d)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: p_new - {}", e),
            })?;

        // Check if we've left the trust region
        let p_new_norm =
            tensor_norm(client, &p_new).map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: p_new norm - {}", e),
            })?;

        if p_new_norm >= trust_radius {
            // Move to boundary along d from p
            let p_boundary = move_to_boundary(client, &p, &d, trust_radius)?;
            let hp = compute_hvp_for_subproblem(client, f, x, &p_boundary)?;
            let pred = compute_predicted_reduction(client, grad, &p_boundary, &hp)?;
            return Ok(SubproblemResult {
                step: p_boundary,
                hits_boundary: true,
                predicted_reduction: pred,
            });
        }

        p = p_new;

        // Update residual: r = r - alpha * H·d
        let alpha_hd =
            client
                .mul_scalar(&hd, alpha)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("steihaug: alpha_hd - {}", e),
                })?;
        r = client
            .sub(&r, &alpha_hd)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: r update - {}", e),
            })?;

        // Update direction
        let r_dot_r_new =
            tensor_dot(client, &r, &r).map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: dot - {}", e),
            })?;
        let beta = r_dot_r_new / r_dot_r;
        r_dot_r = r_dot_r_new;

        let beta_d = client
            .mul_scalar(&d, beta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: beta_d - {}", e),
            })?;
        d = client
            .add(&r, &beta_d)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("steihaug: d update - {}", e),
            })?;
    }

    // Ran out of CG iterations: return current p
    let hp = compute_hvp_for_subproblem(client, f, x, &p)?;
    let pred = compute_predicted_reduction(client, grad, &p, &hp)?;
    let p_norm = tensor_norm(client, &p).map_err(|e| OptimizeError::NumericalError {
        message: format!("steihaug: final p norm - {}", e),
    })?;

    Ok(SubproblemResult {
        step: p,
        hits_boundary: (p_norm - trust_radius).abs() / trust_radius < 0.01,
        predicted_reduction: pred,
    })
}

/// Move from point p along direction d to the trust region boundary.
///
/// Find tau > 0 such that ||p + tau * d|| = trust_radius.
/// This is a quadratic equation in tau: ||p||^2 + 2*tau*(p·d) + tau^2*||d||^2 = delta^2
fn move_to_boundary<R, C>(
    client: &C,
    p: &Tensor<R>,
    d: &Tensor<R>,
    trust_radius: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let p_dot_p = tensor_dot(client, p, p).map_err(|e| OptimizeError::NumericalError {
        message: format!("move_to_boundary: dot - {}", e),
    })?;
    let p_dot_d = tensor_dot(client, p, d).map_err(|e| OptimizeError::NumericalError {
        message: format!("move_to_boundary: dot - {}", e),
    })?;
    let d_dot_d = tensor_dot(client, d, d).map_err(|e| OptimizeError::NumericalError {
        message: format!("move_to_boundary: dot - {}", e),
    })?;

    // Solve: d_dot_d * tau^2 + 2*p_dot_d * tau + (p_dot_p - delta^2) = 0
    let delta_sq = trust_radius * trust_radius;
    let discriminant = p_dot_d * p_dot_d - d_dot_d * (p_dot_p - delta_sq);

    let tau = if discriminant < 0.0 {
        // Shouldn't happen, but fall back to a safe value
        0.0
    } else {
        (-p_dot_d + discriminant.sqrt()) / d_dot_d
    };

    let tau_d = client
        .mul_scalar(d, tau)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("move_to_boundary: tau_d - {}", e),
        })?;
    client
        .add(p, &tau_d)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("move_to_boundary: p + tau_d - {}", e),
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
    fn test_trust_ncg_quadratic() {
        let (device, client) = setup();

        // f(x) = sum(x^2), minimum at origin
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let result = trust_ncg_impl(
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
    fn test_trust_ncg_shifted_quadratic() {
        let (device, client) = setup();

        // f(x) = sum((x - 1)^2), minimum at x = [1, 1]
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0], &[2], &device).unwrap();

        let result = trust_ncg_impl(
            &client,
            |x_var, c| {
                let one = Var::new(
                    Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device).unwrap(),
                    false,
                );
                let diff = numr::autograd::var_sub(x_var, &one, c)?;
                let diff_sq = var_mul(&diff, &diff, c)?;
                var_sum(&diff_sq, &[0], false, c)
            },
            &x0,
            &TrustRegionOptions::default(),
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-10);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 1e-4);
        assert!((sol[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_trust_ncg_sphere() {
        let (device, client) = setup();

        // Sphere function in 10D: f(x) = sum(x^2)
        let initial: Vec<f64> = (0..10).map(|i| (i as f64) - 5.0).collect();
        let x0 = Tensor::<CpuRuntime>::from_slice(&initial, &[10], &device).unwrap();

        let result = trust_ncg_impl(
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
}
