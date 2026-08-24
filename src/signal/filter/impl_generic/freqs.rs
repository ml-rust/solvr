//! Generic implementation of analog frequency response.
//!
//! Computes H(jω) for analog filters in the s-domain.
//!
//! This implementation is fully tensorized - no GPU↔CPU transfers.
//! The computation is embarrassingly parallel across frequency points.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::traits::analog_response::FreqsResult;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute analog frequency response using tensor operations.
///
/// Evaluates H(jω) = B(jω) / A(jω) where:
/// - B(s) = b`[0]`*s^M + b`[1]`*s^(M-1) + ... + b`[M]`
/// - A(s) = a`[0]`*s^N + a`[1]`*s^(N-1) + ... + a`[N]`
/// - s = jω (purely imaginary)
///
/// The computation is fully tensorized:
/// 1. Create power tensors for coefficient indices
/// 2. Compute ω^power for all frequencies and powers (broadcasted)
/// 3. Apply sign patterns based on (jω)^k cycling: [1, j, -1, -j, 1, ...]
/// 4. Sum with coefficient weighting to get B_re, B_im, A_re, A_im
/// 5. Complex division for H = B / A
pub fn freqs_impl<R, C>(
    client: &C,
    b: &Tensor<R>,
    a: &Tensor<R>,
    worN: &Tensor<R>,
    device: &R::Device,
) -> Result<FreqsResult<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let dtype = b.dtype();
    let nb = b.shape()[0];
    let na = a.shape()[0];

    if nb == 0 || na == 0 {
        return Err(Error::InvalidArgument {
            arg: "b/a",
            reason: "Filter coefficients cannot be empty".to_string(),
        });
    }

    // Get leading coefficient for normalization check
    // We need to extract a[0] - use narrow to get first element
    let a0_tensor = a.narrow(0, 0, 1)?;

    // Normalize coefficients: b_norm = b / a[0], a_norm = a / a[0]
    let b_norm = client.div(b, &a0_tensor)?;
    let a_norm = client.div(a, &a0_tensor)?;

    // For B(jω): M = nb - 1, powers are [M, M-1, ..., 1, 0]
    // For A(jω): N = na - 1, powers are [N, N-1, ..., 1, 0]
    let m = nb - 1;
    let n = na - 1;

    // Compute B(jω) = Σ b[k] * (jω)^(M-k)
    // Powers for B: [M, M-1, ..., 0]
    let (b_re, b_im) = evaluate_polynomial_at_jw(client, &b_norm, m, worN, dtype, device)?;

    // Compute A(jω) = Σ a[k] * (jω)^(N-k)
    // Powers for A: [N, N-1, ..., 0]
    let (a_re, a_im) = evaluate_polynomial_at_jw(client, &a_norm, n, worN, dtype, device)?;

    // H = B / A (complex division)
    // H_re = (B_re * A_re + B_im * A_im) / (A_re² + A_im²)
    // H_im = (B_im * A_re - B_re * A_im) / (A_re² + A_im²)
    let a_re_sq = client.mul(&a_re, &a_re)?;
    let a_im_sq = client.mul(&a_im, &a_im)?;
    let denom = client.add(&a_re_sq, &a_im_sq)?;

    // Numerator for real part: B_re * A_re + B_im * A_im
    let num_re_part1 = client.mul(&b_re, &a_re)?;
    let num_re_part2 = client.mul(&b_im, &a_im)?;
    let num_re = client.add(&num_re_part1, &num_re_part2)?;

    // Numerator for imag part: B_im * A_re - B_re * A_im
    let num_im_part1 = client.mul(&b_im, &a_re)?;
    let num_im_part2 = client.mul(&b_re, &a_im)?;
    let num_im = client.sub(&num_im_part1, &num_im_part2)?;

    // Divide (small denominators will give infinity which is correct for poles)
    let h_real = client.div(&num_re, &denom)?;
    let h_imag = client.div(&num_im, &denom)?;

    Ok(FreqsResult {
        w: worN.clone(),
        h_real,
        h_imag,
    })
}

/// Evaluate polynomial at jω using tensor operations.
///
/// For p(jω) = Σ c[k] * (jω)^(M-k) where k = 0..len(c)
///
/// The powers of jω cycle as:
/// - (jω)^0 = 1         → contributes to real
/// - (jω)^1 = jω        → contributes to imag
/// - (jω)^2 = -ω²       → contributes to real (negative)
/// - (jω)^3 = -jω³      → contributes to imag (negative)
/// - (jω)^4 = ω⁴        → contributes to real
/// - ...
///
/// Returns (real_part, imag_part) as tensors of shape [n_freqs].
fn evaluate_polynomial_at_jw<R, C>(
    client: &C,
    coeffs: &Tensor<R>,
    max_power: usize,
    omega: &Tensor<R>,
    dtype: DType,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let n_freqs = omega.shape()[0];
    let n_coeffs = coeffs.shape()[0];

    if n_coeffs == 0 {
        return Ok((
            Tensor::zeros(&[n_freqs], dtype, device)?,
            Tensor::zeros(&[n_freqs], dtype, device)?,
        ));
    }

    // Create power indices: [max_power, max_power-1, ..., max_power-n_coeffs+1]
    // For each coefficient k, the power is max_power - k
    let powers_vec: Vec<f64> = (0..n_coeffs).map(|k| (max_power - k) as f64).collect();
    let powers = Tensor::from_slice(&powers_vec, &[n_coeffs], device)?;

    // Expand omega to [n_freqs, 1] for broadcasting
    let omega_expanded = omega.reshape(&[n_freqs, 1])?;

    // Expand powers to [1, n_coeffs] for broadcasting
    let powers_expanded = powers.reshape(&[1, n_coeffs])?;

    // Compute omega^power: [n_freqs, n_coeffs]
    // This broadcasts omega[i]^power[j] for all combinations
    let omega_pow = client.pow(&omega_expanded, &powers_expanded)?;

    // Create sign patterns based on power % 4
    // Real signs: [1, 0, -1, 0] for powers [0, 1, 2, 3]
    // Imag signs: [0, 1, 0, -1] for powers [0, 1, 2, 3]
    let mut sign_re_vec = Vec::with_capacity(n_coeffs);
    let mut sign_im_vec = Vec::with_capacity(n_coeffs);

    for k in 0..n_coeffs {
        let power = max_power - k;
        match power % 4 {
            0 => {
                sign_re_vec.push(1.0);
                sign_im_vec.push(0.0);
            }
            1 => {
                sign_re_vec.push(0.0);
                sign_im_vec.push(1.0);
            }
            2 => {
                sign_re_vec.push(-1.0);
                sign_im_vec.push(0.0);
            }
            3 => {
                sign_re_vec.push(0.0);
                sign_im_vec.push(-1.0);
            }
            _ => unreachable!(),
        }
    }

    let sign_re = Tensor::from_slice(&sign_re_vec, &[1, n_coeffs], device)?;
    let sign_im = Tensor::from_slice(&sign_im_vec, &[1, n_coeffs], device)?;

    // Expand coefficients to [1, n_coeffs] for broadcasting
    let coeffs_expanded = coeffs.reshape(&[1, n_coeffs])?;

    // Compute weighted terms: coeffs * sign * omega^power
    // Shape: [n_freqs, n_coeffs]
    let weighted_re = client.mul(&coeffs_expanded, &sign_re)?;
    let terms_re = client.mul(&weighted_re, &omega_pow)?;

    let weighted_im = client.mul(&coeffs_expanded, &sign_im)?;
    let terms_im = client.mul(&weighted_im, &omega_pow)?;

    // Sum along coefficient dimension (dim=1) to get [n_freqs]
    let sum_re = client.sum(&terms_re, &[1], false)?;
    let sum_im = client.sum(&terms_im, &[1], false)?;

    Ok((sum_re, sum_im))
}
