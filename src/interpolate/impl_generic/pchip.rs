//! PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) generic implementation.
//!
//! Uses tensor operations for GPU-accelerated computation.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute slopes using the Fritsch-Carlson method with tensor operations.
///
/// This method ensures monotonicity preservation:
/// - If adjacent secants have the same sign, use weighted harmonic mean
/// - If they have opposite signs or either is zero, set slope to zero
///
/// All computation stays on device - no to_vec() calls.
pub fn pchip_slopes<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];
    let device = client.device();

    if n == 2 {
        // For 2 points, slope is just the secant: (y[1] - y[0]) / (x[1] - x[0])
        let x0 = x.narrow(0, 0, 1)?;
        let x1 = x.narrow(0, 1, 1)?;
        let y0 = y.narrow(0, 0, 1)?;
        let y1 = y.narrow(0, 1, 1)?;
        let dx = client.sub(&x1, &x0)?;
        let dy = client.sub(&y1, &y0)?;
        let secant = client.div(&dy, &dx)?;
        return Ok(client.cat(&[&secant, &secant], 0)?);
    }

    // Step 1: Compute secants between adjacent points
    // s[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) for i = 0..n-2
    let x_lo = x.narrow(0, 0, n - 1)?; // x[0..n-1]
    let x_hi = x.narrow(0, 1, n - 1)?; // x[1..n]
    let y_lo = y.narrow(0, 0, n - 1)?; // y[0..n-1]
    let y_hi = y.narrow(0, 1, n - 1)?; // y[1..n]

    let dx = client.sub(&x_hi, &x_lo)?; // h[i] = x[i+1] - x[i]
    let dy = client.sub(&y_hi, &y_lo)?;
    let secants = client.div(&dy, &dx)?; // s[i], shape [n-1]

    // Step 2: Compute interior slopes using Fritsch-Carlson
    // For i = 1..n-2 (interior points):
    //   s0 = secants[i-1], s1 = secants[i]
    //   h0 = dx[i-1], h1 = dx[i]
    //   If s0*s1 <= 0: slope = 0 (monotonicity preservation)
    //   Else: slope = (w1 + w2) / (w1/s0 + w2/s1) where w1 = 2*h1 + h0, w2 = h1 + 2*h0

    let s0 = secants.narrow(0, 0, n - 2)?.contiguous(); // secants[0..n-2]
    let s1 = secants.narrow(0, 1, n - 2)?.contiguous(); // secants[1..n-1]
    let h0 = dx.narrow(0, 0, n - 2)?.contiguous(); // dx[0..n-2]
    let h1 = dx.narrow(0, 1, n - 2)?.contiguous(); // dx[1..n-1]

    let interior_len = n - 2;
    let epsilon_data = vec![1e-14; interior_len];
    let epsilon = Tensor::<R>::from_slice(&epsilon_data, &[interior_len], device);

    // Compute product s0 * s1 to check monotonicity
    let product = client.mul(&s0, &s1)?;
    let product_abs = client.abs(&product)?;

    // Monotonicity indicator: 1 if s0*s1 > 0, 0 otherwise
    // indicator = (product + abs(product)) / (2 * abs(product) + epsilon)
    // When product > 0: (p + p) / (2p + eps) ≈ 1
    // When product <= 0: (p + |p|) / (2|p| + eps) = 0
    let sum_prod = client.add(&product, &product_abs)?;
    let abs_2 = client.mul_scalar(&product_abs, 2.0)?;
    let safe_denom = client.add(&abs_2, &epsilon)?;
    let monotonic_indicator = client.div(&sum_prod, &safe_denom)?;

    // Compute weights: w1 = 2*h1 + h0, w2 = h1 + 2*h0
    let h1_2 = client.mul_scalar(&h1, 2.0)?;
    let h0_2 = client.mul_scalar(&h0, 2.0)?;
    let w1 = client.add(&h1_2, &h0)?; // 2*h1 + h0
    let w2 = client.add(&h1, &h0_2)?; // h1 + 2*h0

    // Weighted harmonic mean with safe division
    // slope = (w1 + w2) / (w1/s0 + w2/s1)
    // To avoid division by zero in s0 or s1, add epsilon to denominators
    let s0_safe = client.add(&s0, &client.mul_scalar(&epsilon, 1e-100)?)?; // tiny perturbation
    let s1_safe = client.add(&s1, &client.mul_scalar(&epsilon, 1e-100)?)?;
    let w_sum = client.add(&w1, &w2)?;
    let w1_over_s0 = client.div(&w1, &s0_safe)?;
    let w2_over_s1 = client.div(&w2, &s1_safe)?;
    let harm_denom = client.add(&w1_over_s0, &w2_over_s1)?;
    let harm_denom_safe = client.add(&harm_denom, &epsilon)?;
    let slope_harmonic = client.div(&w_sum, &harm_denom_safe)?;

    // Interior slopes = harmonic * monotonic_indicator (0 where non-monotonic)
    let interior_slopes = client.mul(&slope_harmonic, &monotonic_indicator)?;

    // Step 3: Compute endpoint slopes using one-sided differences with shape preservation
    // Left endpoint (i=0): use s[0], s[1], h[0], h[1]
    let left_slope = compute_endpoint_slope_tensor(
        client,
        &secants.narrow(0, 0, 1)?,
        &secants.narrow(0, 1, 1)?,
        &dx.narrow(0, 0, 1)?,
        &dx.narrow(0, 1, 1)?,
    )?;

    // Right endpoint (i=n-1): use s[n-2], s[n-3], h[n-2], h[n-3]
    let right_slope = compute_endpoint_slope_tensor(
        client,
        &secants.narrow(0, n - 2, 1)?,
        &secants.narrow(0, n - 3, 1)?,
        &dx.narrow(0, n - 2, 1)?,
        &dx.narrow(0, n - 3, 1)?,
    )?;

    // Concatenate: [left_slope, interior_slopes, right_slope]
    let slopes = client.cat(&[&left_slope, &interior_slopes, &right_slope], 0)?;

    Ok(slopes)
}

/// Compute endpoint slope with shape preservation using tensor ops.
///
/// Formula: d = ((2*h1 + h2) * s1 - h1 * s2) / (h1 + h2)
/// With shape preservation rules:
/// - If sign(d) != sign(s1): return 0
/// - If sign(s1) != sign(s2) && |d| > 3*|s1|: return 3*s1
/// - Otherwise: return d
///
/// Uses smooth indicator functions to avoid where_cond which has issues
/// with non-contiguous tensors.
fn compute_endpoint_slope_tensor<R, C>(
    client: &C,
    s1: &Tensor<R>,
    s2: &Tensor<R>,
    h1: &Tensor<R>,
    h2: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let epsilon = Tensor::<R>::from_slice(&[1e-14], &[1], device);

    // Make inputs contiguous
    let s1 = s1.contiguous();
    let s2 = s2.contiguous();
    let h1 = h1.contiguous();
    let h2 = h2.contiguous();

    // d = ((2*h1 + h2) * s1 - h1 * s2) / (h1 + h2)
    let h1_2 = client.mul_scalar(&h1, 2.0)?;
    let coef = client.add(&h1_2, &h2)?; // 2*h1 + h2
    let term1 = client.mul(&coef, &s1)?; // (2*h1 + h2) * s1
    let term2 = client.mul(&h1, &s2)?; // h1 * s2
    let numer = client.sub(&term1, &term2)?;
    let denom = client.add(&h1, &h2)?;
    let denom_safe = client.add(&denom, &epsilon)?;
    let d = client.div(&numer, &denom_safe)?;

    // Smooth indicator for same sign: (a*b + |a*b|) / (2*|a*b| + eps)
    // Returns ~1 if same sign, ~0 otherwise
    let d_s1_prod = client.mul(&d, &s1)?;
    let d_s1_prod_abs = client.abs(&d_s1_prod)?;
    let d_s1_sum = client.add(&d_s1_prod, &d_s1_prod_abs)?;
    let d_s1_denom = client.add(&client.mul_scalar(&d_s1_prod_abs, 2.0)?, &epsilon)?;
    let d_s1_same_sign = client.div(&d_s1_sum, &d_s1_denom)?; // ~1 if same sign

    let s1_s2_prod = client.mul(&s1, &s2)?;
    let s1_s2_prod_abs = client.abs(&s1_s2_prod)?;
    let s1_s2_sum = client.add(&s1_s2_prod, &s1_s2_prod_abs)?;
    let s1_s2_denom = client.add(&client.mul_scalar(&s1_s2_prod_abs, 2.0)?, &epsilon)?;
    let s1_s2_same_sign = client.div(&s1_s2_sum, &s1_s2_denom)?; // ~1 if same sign

    // s1_s2_diff_sign = 1 - s1_s2_same_sign
    let ones = Tensor::<R>::from_slice(&[1.0], &[1], device);
    let s1_s2_diff_sign = client.sub(&ones, &s1_s2_same_sign)?;

    // Check if |d| > 3*|s1|
    // excess = max(0, |d| - 3*|s1|)
    // indicator = excess / (excess + eps) ≈ 1 if excess > 0
    let d_abs = client.abs(&d)?;
    let s1_abs = client.abs(&s1)?;
    let s1_3 = client.mul_scalar(&s1_abs, 3.0)?;
    let diff = client.sub(&d_abs, &s1_3)?;
    let diff_abs = client.abs(&diff)?;
    let excess = client.mul_scalar(&client.add(&diff, &diff_abs)?, 0.5)?; // max(0, diff)
    let excess_safe = client.add(&excess, &epsilon)?;
    let d_too_large = client.div(&excess, &excess_safe)?; // ~1 if |d| > 3|s1|

    // Rule 2 applies when: s1_s2_diff_sign AND d_too_large
    let rule2_applies = client.mul(&s1_s2_diff_sign, &d_too_large)?;

    // 3*s1 with proper sign: sign(s1) * 3 * |s1| = 3*s1
    let s1_times_3 = client.mul_scalar(&s1, 3.0)?;

    // Blend between d and 3*s1 based on rule2
    // result_base = d * (1 - rule2_applies) + s1_times_3 * rule2_applies
    let one_minus_rule2 = client.sub(&ones, &rule2_applies)?;
    let d_contrib = client.mul(&d, &one_minus_rule2)?;
    let s1_3_contrib = client.mul(&s1_times_3, &rule2_applies)?;
    let result_base = client.add(&d_contrib, &s1_3_contrib)?;

    // Apply rule 1: multiply by d_s1_same_sign (0 if different signs)
    let result = client.mul(&result_base, &d_s1_same_sign)?;

    Ok(result)
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
    fn test_pchip_slopes_linear_data() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0], &[4], &device);

        let slopes = pchip_slopes(&client, &x, &y).unwrap();
        let slopes_vec: Vec<f64> = slopes.to_vec();

        assert_eq!(slopes_vec.len(), 4);
        // For linear data, slopes should all be 2.0
        for &slope in &slopes_vec {
            assert!((slope - 2.0).abs() < 1e-10);
        }
    }
}
