//! Lomb-Scargle periodogram for unevenly sampled data.
//!
//! Uses tensor operations for efficient computation.
//!
//! Note: Lomb-Scargle is inherently O(N*M) where N is signal length and M is
//! number of frequencies. GPU can help parallelize across frequencies, but
//! the algorithm structure limits potential speedup compared to FFT-based methods.
use crate::DType;

use numr::error::{Error, Result};
use numr::ops::{ReduceOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Compute Lomb-Scargle periodogram for unevenly sampled data.
///
/// Uses tensor operations for the core computation. For each frequency,
/// we compute the power using vectorized trig operations.
pub fn lombscargle_impl<R, C>(
    client: &C,
    t: &Tensor<R>,
    x: &Tensor<R>,
    freqs: &Tensor<R>,
    normalize: bool,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + ReduceOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let n_samples = t.shape()[0];
    let n_freqs = freqs.shape()[0];
    let device = t.device();
    let dtype = t.dtype();

    if n_samples != x.shape()[0] {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "t and x must have the same length".to_string(),
        });
    }

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if n_freqs == 0 {
        return Tensor::zeros(&[0], dtype, device);
    }

    // Compute mean and center the data
    let x_mean = client.mean(x, &[0], false)?;
    let x_centered = client.sub(x, &x_mean)?;

    // Compute variance for normalization
    // ddof=0 for population variance
    let x_var_tensor = client.var(&x_centered, &[0], false, 0)?;
    let x_var: f64 = x_var_tensor.item()?;

    // Get frequency values for iteration
    // Note: Lomb-Scargle requires per-frequency tau computation which involves
    // atan2 - this is not easily vectorizable across frequencies without
    // outer product operations. We iterate over frequencies but use tensor
    // ops within each frequency computation.
    let freqs_data: Vec<f64> = freqs.to_vec();
    let mut power_vec = Vec::with_capacity(n_freqs);

    for &freq in &freqs_data {
        let omega = 2.0 * PI * freq;

        // Compute 2*omega*t as a tensor
        let omega_t = client.mul_scalar(t, omega)?;
        let two_omega_t = client.mul_scalar(&omega_t, 2.0)?;

        // Compute sin(2*omega*t) and cos(2*omega*t)
        let sin_2wt = client.sin(&two_omega_t)?;
        let cos_2wt = client.cos(&two_omega_t)?;

        // Sum to compute tau
        let sin_sum = client.sum(&sin_2wt, &[0], false)?;
        let cos_sum = client.sum(&cos_2wt, &[0], false)?;
        let sin_val: f64 = sin_sum.item()?;
        let cos_val: f64 = cos_sum.item()?;
        let tau = sin_val.atan2(cos_val) / (2.0 * omega);

        // Compute omega*(t - tau) = omega*t - omega*tau
        let omega_tau = omega * tau;
        let arg = client.add_scalar(&omega_t, -omega_tau)?;

        // Compute cos(arg) and sin(arg)
        let cos_arg = client.cos(&arg)?;
        let sin_arg = client.sin(&arg)?;

        // Compute sums: x*cos, x*sin, cos², sin²
        let x_cos = client.mul(&x_centered, &cos_arg)?;
        let x_sin = client.mul(&x_centered, &sin_arg)?;
        let cos_sq = client.mul(&cos_arg, &cos_arg)?;
        let sin_sq = client.mul(&sin_arg, &sin_arg)?;

        let cos_sum_tensor = client.sum(&x_cos, &[0], false)?;
        let sin_sum_tensor = client.sum(&x_sin, &[0], false)?;
        let cos2_sum_tensor = client.sum(&cos_sq, &[0], false)?;
        let sin2_sum_tensor = client.sum(&sin_sq, &[0], false)?;

        let cos_sum_val: f64 = cos_sum_tensor.item()?;
        let sin_sum_val: f64 = sin_sum_tensor.item()?;
        let cos2_val: f64 = cos2_sum_tensor.item()?;
        let sin2_val: f64 = sin2_sum_tensor.item()?;

        // Compute power
        let p = if cos2_val.abs() < 1e-30 || sin2_val.abs() < 1e-30 {
            0.0
        } else {
            0.5 * (cos_sum_val * cos_sum_val / cos2_val + sin_sum_val * sin_sum_val / sin2_val)
        };

        // Normalize if requested
        let p = if normalize && x_var > 1e-30 {
            p / x_var
        } else {
            p
        };

        power_vec.push(p);
    }

    Tensor::from_slice(&power_vec, &[n_freqs], device)
}
