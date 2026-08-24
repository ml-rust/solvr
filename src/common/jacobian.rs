//! Autograd-based Jacobian computation shared across solvr.
//!
//! Provides exact Jacobian computation using numr's forward-mode automatic
//! differentiation. This is used by:
//! - `optimize/roots` - Newton's method for root finding
//! - `integrate/ode` - Implicit ODE solvers (BDF, Radau, LSODA)
//!
//! # Why Autograd over Finite Differences?
//!
//! Finite differences have several drawbacks:
//! - Truncation error from finite step size
//! - Numerical instability for ill-conditioned problems
//! - Requires tuning epsilon parameter
//! - O(n) function evaluations per Jacobian
//!
//! Autograd provides:
//! - Exact derivatives (to machine precision)
//! - No epsilon tuning required
//! - More robust for stiff problems
//! - Same O(n) cost but exact results
//!
//! # Usage
//!
//! User must write their function using `DualTensor` and `dual_*` operations:
//!
//! ```no_run
//! # use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
//! # use numr::tensor::Tensor;
//! use numr::autograd::DualTensor;
//! # let device = CpuDevice::new();
//! # let client = CpuClient::new(device.clone());
//! # let y = Tensor::from_slice(&[1.0, 1.0], &[2], &device)?;
//!
//! // f(y) = [y[0]^2 - y[1], y[0] * y[1] - 1]
//! let jacobian = solvr::common::jacobian::jacobian_autograd(&client, |_y: &DualTensor<CpuRuntime>, _c: &CpuClient| {
//!     // Use dual operations for automatic differentiation
//!     unimplemented!()
//! }, &y)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
use crate::DType;

use numr::autograd::{DualTensor, Var, backward, jacobian_forward, jvp, var_mul, var_sum};
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute the Jacobian matrix using forward-mode automatic differentiation.
///
/// For a function F: ℝⁿ → ℝᵐ, computes the m×n Jacobian matrix J where
/// `J[i,j]` = ∂Fᵢ/∂xⱼ.
///
/// This uses numr's forward-mode AD, which computes n JVPs (one per input
/// dimension). For square systems (n = m), this is optimal.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - Function using `DualTensor` and `dual_*` operations
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
///
/// Jacobian matrix of shape `[m, n]`
///
/// # Example
///
/// ```no_run
/// use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
/// use numr::tensor::Tensor;
/// use numr::autograd::DualTensor;
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let x = Tensor::from_slice(&[1.0], &[1], &device)?;
/// // F(x) = x² (element-wise), Jacobian = diag(2x)
/// let jacobian = solvr::common::jacobian::jacobian_autograd(
///     &client,
///     |dual_x: &DualTensor<CpuRuntime>, c: &CpuClient| { /* dual ops */ unimplemented!() },
///     &x,
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn jacobian_autograd<R, C, F>(client: &C, f: F, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    jacobian_forward(f, x, client)
}

/// Compute Jacobian-vector product J @ v using forward-mode AD.
///
/// For a function F: ℝⁿ → ℝᵐ, computes J(x) @ v without forming the full
/// Jacobian matrix. This is useful for:
/// - Newton-Krylov methods (iterative linear solvers)
/// - Large systems where forming J is expensive
/// - Memory-constrained environments
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function using `DualTensor` operations
/// * `x` - Point at which to evaluate
/// * `v` - Vector to multiply with Jacobian
///
/// # Returns
///
/// Tuple of (F(x), J(x) @ v)
pub fn jvp_autograd<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> Result<DualTensor<R>>,
{
    jvp(f, &[x], &[v], client)
}

/// Compute vector-Jacobian product vᵀ @ J using reverse-mode AD.
///
/// For a function F: ℝⁿ → ℝᵐ, computes vᵀ @ J(x) without forming the full
/// Jacobian matrix. This is useful for:
/// - Adjoint sensitivity analysis (computing ∂J/∂p)
/// - Backpropagation through ODE solvers
/// - Computing gradients of scalar loss functions
///
/// # Mathematical Background
///
/// For adjoint ODE: dλ/dt = -λᵀ @ (∂f/∂y)
///
/// At each time step, we need to compute λᵀ @ J_y where J_y = ∂f/∂y.
/// This function computes exactly that using reverse-mode AD in O(1) passes.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function that takes a Var and returns a Var
/// * `x` - Point at which to evaluate
/// * `v` - Vector to left-multiply with Jacobian (the adjoint λ)
///
/// # Returns
///
/// Tuple of (F(x), vᵀ @ J(x))
///
/// # Example
///
/// ```no_run
/// use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
/// use numr::tensor::Tensor;
/// use numr::autograd::Var;
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let x = Tensor::from_slice(&[2.0], &[1], &device)?;
/// let v = Tensor::from_slice(&[1.0], &[1], &device)?;
/// // F(x) = x², at x=[2], v=[1]
/// let (fx, vjp_result) = solvr::common::jacobian::vjp_autograd(
///     &client,
///     |x_var: &Var<CpuRuntime>, c: &CpuClient| { /* var ops */ unimplemented!() },
///     &x,
///     &v,
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn vjp_autograd<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &C) -> Result<Var<R>>,
{
    // Create input variable with grad tracking
    let x_var = Var::new(x.clone(), true);

    // Forward pass
    let y_var = f(&x_var, client)?;
    let fx = y_var.tensor().clone();

    // Compute vᵀ @ y (element-wise multiplication then sum to get scalar loss)
    let v_var = Var::new(v.clone(), false);
    let prod = var_mul(&y_var, &v_var, client)?;
    // Sum over all dimensions to get scalar
    let all_dims: Vec<usize> = (0..prod.tensor().shape().len()).collect();
    let loss = var_sum(&prod, &all_dims, false, client)?;

    // Backward pass to get vᵀ @ J
    let grads = backward(&loss, client)?;

    // Get gradient w.r.t. x
    let vjp_result = match grads.get(x_var.id()) {
        Some(g) => g.clone(),
        None => {
            // If no gradient, return zeros
            Tensor::<R>::zeros(x.shape(), x.dtype(), x.device())?
        }
    };

    Ok((fx, vjp_result))
}

/// Compute the vector-Jacobian product for a function with parameters.
///
/// Computes vᵀ @ (∂f/∂y) and vᵀ @ (∂f/∂p) simultaneously for the adjoint
/// ODE sensitivity method.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function f(t, y, p) returning dy/dt
/// * `t` - Current time (scalar)
/// * `y` - Current state
/// * `p` - Parameters
/// * `v` - Adjoint vector λ
///
/// # Returns
///
/// Tuple of (f(t, y, p), vᵀ @ (∂f/∂y), vᵀ @ (∂f/∂p))
pub fn vjp_with_params<R, C, F>(
    client: &C,
    f: F,
    t: f64,
    y: &Tensor<R>,
    p: &Tensor<R>,
    v: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R>,
    F: Fn(&Var<R>, &Var<R>, &Var<R>, &C) -> Result<Var<R>>,
{
    let device = y.device();

    // Create input variables with grad tracking
    let t_var = Var::new(Tensor::<R>::from_slice(&[t], &[1], device)?, false);
    let y_var = Var::new(y.clone(), true);
    let p_var = Var::new(p.clone(), true);

    // Forward pass
    let f_var = f(&t_var, &y_var, &p_var, client)?;
    let fx = f_var.tensor().clone();

    // Compute vᵀ @ f (element-wise multiplication then sum to get scalar loss)
    let v_var = Var::new(v.clone(), false);
    let prod = var_mul(&f_var, &v_var, client)?;
    // Sum over all dimensions to get scalar
    let all_dims: Vec<usize> = (0..prod.tensor().shape().len()).collect();
    let loss = var_sum(&prod, &all_dims, false, client)?;

    // Backward pass
    let grads = backward(&loss, client)?;

    // Get gradients w.r.t. y and p
    let vjp_y = match grads.get(y_var.id()) {
        Some(g) => g.clone(),
        None => Tensor::<R>::zeros(y.shape(), y.dtype(), device)?,
    };

    let vjp_p = match grads.get(p_var.id()) {
        Some(g) => g.clone(),
        None => Tensor::<R>::zeros(p.shape(), p.dtype(), device)?,
    };

    Ok((fx, vjp_y, vjp_p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul, dual_mul_scalar};
    use numr::autograd::var_mul_scalar;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_jacobian_autograd_linear() {
        let (device, client) = setup();

        // F(x) = 2x, Jacobian = 2I
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let jacobian =
            jacobian_autograd(&client, |dual_x, c| dual_mul_scalar(dual_x, 2.0, c), &x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Diagonal should be 2, off-diagonal 0
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 2.0).abs() < 1e-10);
        assert!((j[8] - 2.0).abs() < 1e-10);
        assert!(j[1].abs() < 1e-10);
        assert!(j[2].abs() < 1e-10);
        assert!(j[3].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_autograd_quadratic() {
        let (device, client) = setup();

        // F(x) = x², Jacobian = diag(2x)
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let jacobian =
            jacobian_autograd(&client, |dual_x, c| dual_mul(dual_x, dual_x, c), &x).unwrap();

        let j: Vec<f64> = jacobian.to_vec();

        // Diagonal should be [2, 4, 6]
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 4.0).abs() < 1e-10);
        assert!((j[8] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_jvp_autograd() {
        let (device, client) = setup();

        // F(x) = x², at x=[2], v=[1]
        // F(x) = 4, J @ v = 2*2*1 = 4
        let x = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();

        let (fx, jv) = jvp_autograd(
            &client,
            |inputs, c| {
                let x = &inputs[0];
                dual_mul(x, x, c)
            },
            &x,
            &v,
        )
        .unwrap();

        assert!((fx.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
        assert!((jv.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_vjp_autograd_simple() {
        let (device, client) = setup();

        // F(x) = x², at x=[2], v=[1]
        // vᵀ @ J = 1 * 2*2 = 4
        let x = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();

        let (fx, vjp_result) =
            vjp_autograd(&client, |x_var, c| var_mul(x_var, x_var, c), &x, &v).unwrap();

        assert!((fx.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
        assert!((vjp_result.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_vjp_autograd_linear() {
        let (device, client) = setup();

        // F(x) = 2x, Jacobian = 2I
        // vᵀ @ J = [1, 1, 1] @ 2I = [2, 2, 2]
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device).unwrap();

        let (fx, vjp_result) =
            vjp_autograd(&client, |x_var, c| var_mul_scalar(x_var, 2.0, c), &x, &v).unwrap();

        let fx_vals: Vec<f64> = fx.to_vec();
        assert!((fx_vals[0] - 2.0).abs() < 1e-10);
        assert!((fx_vals[1] - 4.0).abs() < 1e-10);
        assert!((fx_vals[2] - 6.0).abs() < 1e-10);

        let vjp_vals: Vec<f64> = vjp_result.to_vec();
        assert!((vjp_vals[0] - 2.0).abs() < 1e-10);
        assert!((vjp_vals[1] - 2.0).abs() < 1e-10);
        assert!((vjp_vals[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vjp_with_params() {
        let (device, client) = setup();

        // f(t, y, p) = p * y (scalar multiplication)
        // ∂f/∂y = p, ∂f/∂p = y
        // vᵀ @ (∂f/∂y) = v * p, vᵀ @ (∂f/∂p) = v * y
        let y = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device).unwrap();
        let p = Tensor::<CpuRuntime>::from_slice(&[3.0f64], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();

        let (fx, vjp_y, vjp_p) = vjp_with_params(
            &client,
            |_t, y_var, p_var, c| var_mul(p_var, y_var, c),
            0.0,
            &y,
            &p,
            &v,
        )
        .unwrap();

        // f = p * y = 3 * 2 = 6
        assert!((fx.to_vec::<f64>()[0] - 6.0).abs() < 1e-10);

        // vᵀ @ (∂f/∂y) = v * p = 1 * 3 = 3
        assert!(
            (vjp_y.to_vec::<f64>()[0] - 3.0).abs() < 1e-10,
            "vjp_y = {}",
            vjp_y.to_vec::<f64>()[0]
        );

        // vᵀ @ (∂f/∂p) = v * y = 1 * 2 = 2
        assert!(
            (vjp_p.to_vec::<f64>()[0] - 2.0).abs() < 1e-10,
            "vjp_p = {}",
            vjp_p.to_vec::<f64>()[0]
        );
    }
}
