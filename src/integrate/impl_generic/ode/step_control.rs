//! Fully device-resident step size control for adaptive ODE methods.
//!
//! All computations stay on device - no scalar transfers during stepping.
use crate::DType;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute normalized error as a scalar tensor (stays on device).
///
/// Returns error tensor of shape `[1]` representing sqrt(sum((y_err/sc)^2)/n).
pub fn compute_error<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_err: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n = y_new.shape()[0] as f64;
    let device = client.device();

    // sc = atol + rtol * max(|y_old|, |y_new|)
    let y_old_abs = client.abs(y_old)?;
    let y_new_abs = client.abs(y_new)?;
    let y_max = client.maximum(&y_old_abs, &y_new_abs)?;
    let sc = client.add_scalar(&client.mul_scalar(&y_max, rtol)?, atol)?;

    // err_scaled = y_err / sc
    let err_scaled = client.div(y_err, &sc)?;
    let err_sq = client.mul(&err_scaled, &err_scaled)?;

    // sum_sq = sum(err_sq), then sqrt(sum_sq / n)
    let sum_sq = client.sum(&err_sq, &[0], false)?;

    // Compute sqrt(sum_sq / n) on device
    let n_tensor = Tensor::<R>::from_slice(&[n], &[1], device)?;
    let mean_sq = client.div(&sum_sq, &n_tensor)?;

    // sqrt via x^0.5
    client.pow_scalar(&mean_sq, 0.5)
}

/// Compute step size factor as a scalar tensor (stays on device).
///
/// Returns factor = safety * (1/error)^(1/(order+1)), clamped to [min_factor, max_factor].
pub fn compute_step_factor<R, C>(
    client: &C,
    error: &Tensor<R>,
    order: usize,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let exponent = 1.0 / (order as f64 + 1.0);

    // inv_error = 1 / (error + epsilon) to avoid division by zero
    let epsilon = Tensor::<R>::from_slice(&[1e-10], &[1], device)?;
    let error_safe = client.add(error, &epsilon)?;
    let one = Tensor::<R>::from_slice(&[1.0], &[1], device)?;
    let inv_error = client.div(&one, &error_safe)?;

    // factor = safety * inv_error^exponent
    let power = client.pow_scalar(&inv_error, exponent)?;
    let factor = client.mul_scalar(&power, safety)?;

    // Clamp to [min_factor, max_factor]
    let min_t = Tensor::<R>::from_slice(&[min_factor], &[1], device)?;
    let max_t = Tensor::<R>::from_slice(&[max_factor], &[1], device)?;
    let factor_clamped = client.minimum(&client.maximum(&factor, &min_t)?, &max_t)?;

    Ok(factor_clamped)
}

/// Compute acceptance indicator as a scalar tensor (stays on device).
///
/// Returns ~1.0 if error <= 1.0, ~0.0 otherwise.
/// Uses smooth step function for numerical stability.
pub fn compute_acceptance<R, C>(client: &C, error: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();

    // diff = 1.0 - error (positive if error < 1)
    let one = Tensor::<R>::from_slice(&[1.0], &[1], device)?;
    let diff = client.sub(&one, error)?;

    // Smooth indicator: (diff + |diff|) / (2 * |diff| + epsilon)
    // This gives ~1 if diff > 0 (error < 1), ~0 if diff < 0 (error > 1)
    let diff_abs = client.abs(&diff)?;
    let epsilon = Tensor::<R>::from_slice(&[1e-14], &[1], device)?;
    let numerator = client.add(&diff, &diff_abs)?;
    let denominator = client.add(&client.mul_scalar(&diff_abs, 2.0)?, &epsilon)?;

    client.div(&numerator, &denominator)
}

/// Conditionally update state based on acceptance indicator.
///
/// Returns y_new if accepted (~1), y_old otherwise (~0).
/// y_result = accepted * y_new + (1 - accepted) * y_old
pub fn conditional_update<R, C>(
    client: &C,
    y_old: &Tensor<R>,
    y_new: &Tensor<R>,
    accepted: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let one = Tensor::<R>::from_slice(&[1.0], &[1], device)?;

    // one_minus_accepted = 1 - accepted
    let one_minus = client.sub(&one, accepted)?;

    // result = accepted * y_new + (1 - accepted) * y_old
    let term1 = client.mul(y_new, accepted)?;
    let term2 = client.mul(y_old, &one_minus)?;

    client.add(&term1, &term2)
}

/// Compute initial step size as a scalar tensor (stays on device).
///
/// Uses Hairer & Wanner algorithm but keeps all computation on device.
#[allow(clippy::too_many_arguments)]
pub fn compute_initial_step<R, C, F>(
    client: &C,
    f: &F,
    t0: &Tensor<R>,
    y0: &Tensor<R>,
    f0: &Tensor<R>,
    order: usize,
    rtol: f64,
    atol: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n = y0.shape()[0] as f64;
    let n_tensor = Tensor::<R>::from_slice(&[n], &[1], device)?;

    // Compute scaling: sc = atol + rtol * |y0|
    let y0_abs = client.abs(y0)?;
    let sc = client.add_scalar(&client.mul_scalar(&y0_abs, rtol)?, atol)?;

    // d0 = ||y0 / sc|| / sqrt(n)
    let y0_scaled = client.div(y0, &sc)?;
    let y0_scaled_sq = client.mul(&y0_scaled, &y0_scaled)?;
    let d0_sq = client.sum(&y0_scaled_sq, &[0], false)?;
    let d0_sq_n = client.div(&d0_sq, &n_tensor)?;
    let d0 = client.pow_scalar(&d0_sq_n, 0.5)?;

    // d1 = ||f0 / sc|| / sqrt(n)
    let f0_scaled = client.div(f0, &sc)?;
    let f0_scaled_sq = client.mul(&f0_scaled, &f0_scaled)?;
    let d1_sq = client.sum(&f0_scaled_sq, &[0], false)?;
    let d1_sq_n = client.div(&d1_sq, &n_tensor)?;
    let d1 = client.pow_scalar(&d1_sq_n, 0.5)?;

    // h0 = 0.01 * d0 / (d1 + epsilon)
    let epsilon = Tensor::<R>::from_slice(&[1e-10], &[1], device)?;
    let d1_safe = client.add(&d1, &epsilon)?;
    let ratio = client.div(&d0, &d1_safe)?;
    let h0 = client.mul_scalar(&ratio, 0.01)?;

    // Clamp h0 to reasonable range
    let min_h = Tensor::<R>::from_slice(&[1e-6], &[1], device)?;
    let max_h = Tensor::<R>::from_slice(&[0.1], &[1], device)?;
    let h0_clamped = client.minimum(&client.maximum(&h0, &min_h)?, &max_h)?;

    // Euler step: y1 = y0 + h0 * f0
    let h0_f0 = client.mul(&h0_clamped, f0)?;
    let y1 = client.add(y0, &h0_f0)?;

    // f1 = f(t0 + h0, y1)
    let t1 = client.add(t0, &h0_clamped)?;
    let f1 = f(&t1, &y1)?;

    // d2 = ||f1 - f0|| / (h0 * sc) / sqrt(n)
    let df = client.sub(&f1, f0)?;
    let df_scaled = client.div(&df, &sc)?;
    let df_sq = client.mul(&df_scaled, &df_scaled)?;
    let d2_sq = client.sum(&df_sq, &[0], false)?;
    let d2_sq_n = client.div(&d2_sq, &n_tensor)?;
    let d2_sqrt = client.pow_scalar(&d2_sq_n, 0.5)?;
    let d2 = client.div(&d2_sqrt, &h0_clamped)?;

    // h1 = (0.01 / max(d1, d2))^(1/(order+1))
    let d_max = client.maximum(&d1, &d2)?;
    let d_max_safe = client.add(&d_max, &epsilon)?;
    let point_01 = Tensor::<R>::from_slice(&[0.01], &[1], device)?;
    let ratio2 = client.div(&point_01, &d_max_safe)?;
    let exponent = 1.0 / (order as f64 + 1.0);
    let h1 = client.pow_scalar(&ratio2, exponent)?;

    // Final: min(h0, h1), clamped
    let h_final = client.minimum(&h0_clamped, &h1)?;
    client.minimum(&client.maximum(&h_final, &min_h)?, &max_h)
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
    fn test_compute_error() {
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[1.1, 2.1, 3.1], &[3], &device).unwrap();
        let y_err =
            Tensor::<CpuRuntime>::from_slice(&[0.001, 0.001, 0.001], &[3], &device).unwrap();

        // With rtol=1e-3, atol=1e-3: sc ≈ 0.001 + 0.001 * 3.1 ≈ 0.004
        // err_scaled = 0.001 / 0.004 ≈ 0.25
        // error = sqrt(mean(0.25^2)) ≈ 0.25 < 1
        let error = compute_error(&client, &y_new, &y_err, &y_old, 1e-3, 1e-3).unwrap();
        let error_val: Vec<f64> = error.to_vec();

        assert!(error_val[0] > 0.0);
        assert!(error_val[0] < 1.0); // Small error should be < 1
    }

    #[test]
    fn test_compute_acceptance() {
        let (device, client) = setup();

        // Error < 1 should give acceptance ~1
        let error_small = Tensor::<CpuRuntime>::from_slice(&[0.5], &[1], &device).unwrap();
        let accept = compute_acceptance(&client, &error_small).unwrap();
        let accept_val: Vec<f64> = accept.to_vec();
        assert!(accept_val[0] > 0.9);

        // Error > 1 should give acceptance ~0
        let error_large = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device).unwrap();
        let reject = compute_acceptance(&client, &error_large).unwrap();
        let reject_val: Vec<f64> = reject.to_vec();
        assert!(reject_val[0] < 0.1);
    }

    #[test]
    fn test_step_factor() {
        let (device, client) = setup();

        // Small error should give large factor (clamped to max)
        let error_small = Tensor::<CpuRuntime>::from_slice(&[0.01], &[1], &device).unwrap();
        let factor = compute_step_factor(&client, &error_small, 4, 0.9, 0.2, 10.0).unwrap();
        let factor_val: Vec<f64> = factor.to_vec();
        assert!(factor_val[0] >= 1.0);

        // Large error should give small factor
        let error_large = Tensor::<CpuRuntime>::from_slice(&[10.0], &[1], &device).unwrap();
        let factor2 = compute_step_factor(&client, &error_large, 4, 0.9, 0.2, 10.0).unwrap();
        let factor2_val: Vec<f64> = factor2.to_vec();
        assert!(factor2_val[0] <= 1.0);
    }

    #[test]
    fn test_conditional_update() {
        let (device, client) = setup();

        let y_old = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0], &[2], &device).unwrap();

        // Accepted = 1 should give y_new
        let accepted = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let result = conditional_update(&client, &y_old, &y_new, &accepted).unwrap();
        let result_val: Vec<f64> = result.to_vec();
        assert!((result_val[0] - 10.0).abs() < 1e-10);

        // Accepted = 0 should give y_old
        let rejected = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();
        let result2 = conditional_update(&client, &y_old, &y_new, &rejected).unwrap();
        let result2_val: Vec<f64> = result2.to_vec();
        assert!((result2_val[0] - 1.0).abs() < 1e-10);
    }
}
