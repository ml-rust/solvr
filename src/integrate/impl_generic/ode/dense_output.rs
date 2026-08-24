//! Dense output for ODE solvers using Hermite interpolation.
//!
//! Provides the ability to evaluate the solution at any point within
//! a completed step, which is essential for accurate event detection.
//!
//! # Mathematical Background
//!
//! For a completed step from (t_old, y_old) to (t_new, y_new), with
//! derivatives f_old = f(t_old, y_old) and f_new = f(t_new, y_new),
//! cubic Hermite interpolation gives:
//!
//! ```text
//! θ = (t - t_old) / h,  h = t_new - t_old
//! y(t) ≈ (1-θ)·y_old + θ·y_new + θ(θ-1)·[(1-2θ)·(y_new - y_old) + (θ-1)·h·f_old + θ·h·f_new]
//! ```
//!
//! This is equivalent to the Hermite basis:
//! ```text
//! y(t) = H₀₀(θ)·y_old + H₁₀(θ)·y_new + H₀₁(θ)·h·f_old + H₁₁(θ)·h·f_new
//! ```
use crate::DType;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Information needed for dense output within a step.
///
/// Stores the endpoints and derivatives for Hermite interpolation.
#[derive(Debug, Clone)]
pub struct DenseOutputStep<R: Runtime<DType = DType>> {
    /// Start time of the step.
    pub t_old: f64,

    /// End time of the step.
    pub t_new: f64,

    /// State at start of step.
    pub y_old: Tensor<R>,

    /// State at end of step.
    pub y_new: Tensor<R>,

    /// Derivative at start of step: f(t_old, y_old).
    pub f_old: Tensor<R>,

    /// Derivative at end of step: f(t_new, y_new).
    pub f_new: Tensor<R>,
}

impl<R: Runtime<DType = DType>> DenseOutputStep<R> {
    /// Create a new dense output step.
    pub fn new(
        t_old: f64,
        t_new: f64,
        y_old: Tensor<R>,
        y_new: Tensor<R>,
        f_old: Tensor<R>,
        f_new: Tensor<R>,
    ) -> Self {
        Self {
            t_old,
            t_new,
            y_old,
            y_new,
            f_old,
            f_new,
        }
    }

    /// Get the step size.
    pub fn h(&self) -> f64 {
        self.t_new - self.t_old
    }

    /// Check if a time is within this step.
    pub fn contains(&self, t: f64) -> bool {
        t >= self.t_old && t <= self.t_new
    }

    /// Compute the normalized time parameter θ ∈ [0, 1].
    pub fn theta(&self, t: f64) -> f64 {
        if self.h().abs() < 1e-15 {
            0.0
        } else {
            (t - self.t_old) / self.h()
        }
    }
}

/// Evaluate the solution at an arbitrary time within a step using Hermite interpolation.
///
/// Uses cubic Hermite interpolation which provides O(h⁴) accuracy,
/// matching the local error order of RK45.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `step` - Dense output step containing endpoint data
/// * `t` - Time at which to evaluate (must be in [t_old, t_new])
///
/// # Returns
///
/// Interpolated state y(t) as a tensor.
///
/// # Algorithm
///
/// Using the normalized parameter θ = (t - t_old) / h:
///
/// ```text
/// y(θ) = y_old + θ * h * f_old
///      + θ² * (3*(y_new - y_old) - h*(2*f_old + f_new))
///      + θ³ * (-2*(y_new - y_old) + h*(f_old + f_new))
/// ```
pub fn dense_eval<R, C>(client: &C, step: &DenseOutputStep<R>, t: f64) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let h = step.h();
    let theta = step.theta(t);

    // Edge cases
    if theta <= 0.0 {
        return Ok(step.y_old.clone());
    }
    if theta >= 1.0 {
        return Ok(step.y_new.clone());
    }

    // Precompute common terms
    // dy = y_new - y_old
    let dy = client.sub(&step.y_new, &step.y_old)?;

    // h_f_old = h * f_old
    let h_f_old = client.mul_scalar(&step.f_old, h)?;

    // h_f_new = h * f_new
    let h_f_new = client.mul_scalar(&step.f_new, h)?;

    // Hermite interpolation coefficients for y(θ)
    // Using the form: y = y_old + θ·h·f_old + θ²·a + θ³·b
    // where:
    //   a = 3·dy - h·(2·f_old + f_new)
    //   b = -2·dy + h·(f_old + f_new)

    // a = 3*dy - 2*h*f_old - h*f_new
    let three_dy = client.mul_scalar(&dy, 3.0)?;
    let two_h_f_old = client.mul_scalar(&h_f_old, 2.0)?;
    let a = client.sub(&client.sub(&three_dy, &two_h_f_old)?, &h_f_new)?;

    // b = -2*dy + h*f_old + h*f_new
    let neg_two_dy = client.mul_scalar(&dy, -2.0)?;
    let b = client.add(&client.add(&neg_two_dy, &h_f_old)?, &h_f_new)?;

    // y(θ) = y_old + θ*h*f_old + θ²*a + θ³*b
    let theta_sq = theta * theta;
    let theta_cu = theta_sq * theta;

    let term1 = client.mul_scalar(&h_f_old, theta)?;
    let term2 = client.mul_scalar(&a, theta_sq)?;
    let term3 = client.mul_scalar(&b, theta_cu)?;

    let result = client.add(&step.y_old, &term1)?;
    let result = client.add(&result, &term2)?;
    client.add(&result, &term3)
}

/// Evaluate the derivative at an arbitrary time within a step.
///
/// Differentiating the Hermite interpolation:
/// ```text
/// y'(θ) = (1/h) * (h*f_old + 2*θ*a + 3*θ²*b)
/// ```
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `step` - Dense output step containing endpoint data
/// * `t` - Time at which to evaluate
///
/// # Returns
///
/// Interpolated derivative y'(t) as a tensor.
pub fn dense_eval_derivative<R, C>(
    client: &C,
    step: &DenseOutputStep<R>,
    t: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let h = step.h();
    let theta = step.theta(t);

    // Edge cases
    if theta <= 0.0 {
        return Ok(step.f_old.clone());
    }
    if theta >= 1.0 {
        return Ok(step.f_new.clone());
    }

    // Precompute common terms (same as in dense_eval)
    let dy = client.sub(&step.y_new, &step.y_old)?;
    let h_f_old = client.mul_scalar(&step.f_old, h)?;
    let h_f_new = client.mul_scalar(&step.f_new, h)?;

    // a = 3*dy - 2*h*f_old - h*f_new
    let three_dy = client.mul_scalar(&dy, 3.0)?;
    let two_h_f_old = client.mul_scalar(&h_f_old, 2.0)?;
    let a = client.sub(&client.sub(&three_dy, &two_h_f_old)?, &h_f_new)?;

    // b = -2*dy + h*f_old + h*f_new
    let neg_two_dy = client.mul_scalar(&dy, -2.0)?;
    let b = client.add(&client.add(&neg_two_dy, &h_f_old)?, &h_f_new)?;

    // y'(θ)*h = h*f_old + 2*θ*a + 3*θ²*b
    // y'(θ) = f_old + (2*θ/h)*a + (3*θ²/h)*b
    let theta_sq = theta * theta;

    let term1 = client.mul_scalar(&a, 2.0 * theta / h)?;
    let term2 = client.mul_scalar(&b, 3.0 * theta_sq / h)?;

    let result = client.add(&step.f_old, &term1)?;
    client.add(&result, &term2)
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
    fn test_dense_eval_endpoints() {
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);

        // At t_old
        let y_at_start = dense_eval(&client, &step, 0.0).unwrap();
        assert!((y_at_start.to_vec::<f64>()[0] - 0.0).abs() < 1e-10);

        // At t_new
        let y_at_end = dense_eval(&client, &step, 1.0).unwrap();
        assert!((y_at_end.to_vec::<f64>()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_eval_linear() {
        // For dy/dt = 1, y(0) = 0, exact solution is y(t) = t
        // Hermite should reproduce this exactly
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);

        // Test at several interior points
        for t in [0.25, 0.5, 0.75] {
            let y_interp = dense_eval(&client, &step, t).unwrap();
            assert!(
                (y_interp.to_vec::<f64>()[0] - t).abs() < 1e-10,
                "Failed at t = {}: got {}",
                t,
                y_interp.to_vec::<f64>()[0]
            );
        }
    }

    #[test]
    fn test_dense_eval_quadratic() {
        // For dy/dt = 2t, y(0) = 0, exact solution is y(t) = t²
        // Over [0, 1]: y(0) = 0, y(1) = 1, f(0) = 0, f(1) = 2
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap(); // f(0) = 0
        let f_new = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device).unwrap(); // f(1) = 2

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);

        // Test at midpoint: exact y(0.5) = 0.25
        let y_mid = dense_eval(&client, &step, 0.5).unwrap();
        assert!(
            (y_mid.to_vec::<f64>()[0] - 0.25).abs() < 1e-10,
            "Expected 0.25, got {}",
            y_mid.to_vec::<f64>()[0]
        );

        // Cubic Hermite reproduces quadratics exactly
        for t in [0.25, 0.5, 0.75] {
            let y_interp = dense_eval(&client, &step, t).unwrap();
            let exact = t * t;
            assert!(
                (y_interp.to_vec::<f64>()[0] - exact).abs() < 1e-10,
                "Failed at t = {}: got {}, expected {}",
                t,
                y_interp.to_vec::<f64>()[0],
                exact
            );
        }
    }

    #[test]
    fn test_dense_eval_derivative() {
        // For dy/dt = 2t: check that derivative interpolation works
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);

        // At endpoints
        let f_at_start = dense_eval_derivative(&client, &step, 0.0).unwrap();
        assert!((f_at_start.to_vec::<f64>()[0] - 0.0).abs() < 1e-10);

        let f_at_end = dense_eval_derivative(&client, &step, 1.0).unwrap();
        assert!((f_at_end.to_vec::<f64>()[0] - 2.0).abs() < 1e-10);

        // At midpoint: exact f(0.5) = 2*0.5 = 1.0
        let f_mid = dense_eval_derivative(&client, &step, 0.5).unwrap();
        assert!(
            (f_mid.to_vec::<f64>()[0] - 1.0).abs() < 1e-10,
            "Expected 1.0, got {}",
            f_mid.to_vec::<f64>()[0]
        );
    }

    #[test]
    fn test_dense_output_step_methods() {
        let (device, _client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 2.0, y_old, y_new, f_old, f_new);

        assert!((step.h() - 2.0).abs() < 1e-10);
        assert!(step.contains(1.0));
        assert!(step.contains(0.0));
        assert!(step.contains(2.0));
        assert!(!step.contains(-0.1));
        assert!(!step.contains(2.1));
        assert!((step.theta(1.0) - 0.5).abs() < 1e-10);
    }
}
