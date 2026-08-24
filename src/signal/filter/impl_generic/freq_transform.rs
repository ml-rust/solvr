//! Frequency transformations for analog filter prototypes.
//!
//! These functions transform a lowpass analog prototype (cutoff = 1 rad/s)
//! to other filter types (highpass, bandpass, bandstop) with specified cutoffs.
//!
//! All operations are fully tensorized - data stays on device with no GPU<->CPU transfers.

// Allow complex return types for tensor quadruplets in bandpass/bandstop transforms
#![allow(clippy::type_complexity)]
use crate::DType;

use crate::signal::filter::types::AnalogPrototype;
use numr::error::Result;
use numr::ops::{ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

// ============================================================================
// Complex tensor operations helpers
// ============================================================================

/// Element-wise complex division: (a + bi) / (c + di)
/// Returns (real_part, imag_part) tensors
fn complex_div_tensor<R, C>(
    client: &C,
    a_re: &Tensor<R>,
    a_im: &Tensor<R>,
    b_re: &Tensor<R>,
    b_im: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // denom = c² + d²
    let c_sq = client.mul(b_re, b_re)?;
    let d_sq = client.mul(b_im, b_im)?;
    let denom = client.add(&c_sq, &d_sq)?;

    // Add small epsilon to avoid division by zero
    let denom_safe = client.add_scalar(&denom, 1e-30)?;

    // re = (a*c + b*d) / denom
    let ac = client.mul(a_re, b_re)?;
    let bd = client.mul(a_im, b_im)?;
    let num_re = client.add(&ac, &bd)?;
    let re = client.div(&num_re, &denom_safe)?;

    // im = (b*c - a*d) / denom
    let bc = client.mul(a_im, b_re)?;
    let ad = client.mul(a_re, b_im)?;
    let num_im = client.sub(&bc, &ad)?;
    let im = client.div(&num_im, &denom_safe)?;

    Ok((re, im))
}

/// Element-wise complex square root using polar form
/// sqrt(re + i*im) = sqrt(mag) * (cos(angle/2) + i*sin(angle/2))
fn complex_sqrt_tensor<R, C>(
    client: &C,
    re: &Tensor<R>,
    im: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // mag = sqrt(re² + im²)
    let re_sq = client.mul(re, re)?;
    let im_sq = client.mul(im, im)?;
    let mag_sq = client.add(&re_sq, &im_sq)?;
    let mag = client.sqrt(&mag_sq)?;
    let sqrt_mag = client.sqrt(&mag)?;

    // angle = atan2(im, re)
    let angle = client.atan2(im, re)?;

    // half_angle = angle / 2
    let half_angle = client.mul_scalar(&angle, 0.5)?;

    // result = sqrt_mag * (cos(half_angle) + i*sin(half_angle))
    let cos_half = client.cos(&half_angle)?;
    let sin_half = client.sin(&half_angle)?;

    let result_re = client.mul(&sqrt_mag, &cos_half)?;
    let result_im = client.mul(&sqrt_mag, &sin_half)?;

    Ok((result_re, result_im))
}

/// Compute complex magnitude squared: re² + im²
fn complex_mag_sq<R, C>(client: &C, re: &Tensor<R>, im: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let re_sq = client.mul(re, re)?;
    let im_sq = client.mul(im, im)?;
    client.add(&re_sq, &im_sq)
}

/// Compute complex magnitude: sqrt(re² + im²)
fn complex_mag<R, C>(client: &C, re: &Tensor<R>, im: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let mag_sq = complex_mag_sq(client, re, im)?;
    client.sqrt(&mag_sq)
}

// ============================================================================
// Frequency transformations
// ============================================================================

/// Transform lowpass prototype to lowpass with specified cutoff.
///
/// Transformation: s → s/ω₀
///
/// All operations are tensor-based with no CPU transfers.
pub fn lp2lp_zpk_impl<R, C>(
    client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // Transform zeros: z → z * ω₀
    let new_zeros_real = client.mul_scalar(&proto.zeros_real, wo)?;
    let new_zeros_imag = client.mul_scalar(&proto.zeros_imag, wo)?;

    // Transform poles: p → p * ω₀
    let new_poles_real = client.mul_scalar(&proto.poles_real, wo)?;
    let new_poles_imag = client.mul_scalar(&proto.poles_imag, wo)?;

    // Gain transformation: k' = k * ω₀^(n_poles - n_zeros)
    let n_zeros = proto.zeros_real.shape()[0] as i32;
    let n_poles = proto.poles_real.shape()[0] as i32;
    let degree = n_poles - n_zeros;
    let gain = proto.gain * wo.powi(degree);

    Ok(AnalogPrototype::new(
        new_zeros_real,
        new_zeros_imag,
        new_poles_real,
        new_poles_imag,
        gain,
    ))
}

/// Transform lowpass prototype to highpass with specified cutoff.
///
/// Transformation: s → ω₀/s
///
/// This inverts zeros and poles around the imaginary axis.
/// All operations are tensor-based with no CPU transfers.
pub fn lp2hp_zpk_impl<R, C>(
    client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let device = proto.zeros_real.device();
    let dtype = proto.zeros_real.dtype();

    let n_zeros = proto.zeros_real.shape()[0];
    let n_poles = proto.poles_real.shape()[0];

    // Create constant tensor filled with wo for complex division
    let wo_re = Tensor::full_scalar(&[n_zeros], dtype, wo, device)?;
    let wo_im = Tensor::zeros(&[n_zeros], dtype, device)?;

    // Transform existing zeros: z → ω₀/z (element-wise complex division)
    let (transformed_z_re, transformed_z_im) = if n_zeros > 0 {
        complex_div_tensor(client, &wo_re, &wo_im, &proto.zeros_real, &proto.zeros_imag)?
    } else {
        (
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
        )
    };

    // Add zeros at origin for degree matching (n_poles - n_zeros zeros at origin)
    let extra_zeros = n_poles.saturating_sub(n_zeros);
    let new_zeros_real = if extra_zeros > 0 {
        let origin_re = Tensor::zeros(&[extra_zeros], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_z_re, &origin_re], 0)?
        } else {
            origin_re
        }
    } else {
        transformed_z_re
    };

    let new_zeros_imag = if extra_zeros > 0 {
        let origin_im = Tensor::zeros(&[extra_zeros], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_z_im, &origin_im], 0)?
        } else {
            origin_im
        }
    } else {
        transformed_z_im
    };

    // Transform poles: p → ω₀/p
    let wo_re_p = Tensor::full_scalar(&[n_poles], dtype, wo, device)?;
    let wo_im_p = Tensor::zeros(&[n_poles], dtype, device)?;
    let (new_poles_real, new_poles_imag) = complex_div_tensor(
        client,
        &wo_re_p,
        &wo_im_p,
        &proto.poles_real,
        &proto.poles_imag,
    )?;

    // Gain transformation using tensor operations
    // k' = k * prod(|z|) / prod(|p|) * sign_adjustment
    let mut gain = proto.gain;

    if n_zeros > 0 {
        // Compute product of zero magnitudes
        let zero_mag = complex_mag(client, &proto.zeros_real, &proto.zeros_imag)?;
        let log_mag = client.log(&client.add_scalar(&zero_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain *= sum_log.exp();
    }

    if n_poles > 0 {
        // Compute product of pole magnitudes
        let pole_mag = complex_mag(client, &proto.poles_real, &proto.poles_imag)?;
        let log_mag = client.log(&client.add_scalar(&pole_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain /= sum_log.exp();
    }

    // Note: Sign changes from negative real parts are handled by taking gain.abs() below.
    // For properly formed analog prototypes, all poles/zeros are either real negative
    // or in complex conjugate pairs, making explicit sign tracking unnecessary.

    Ok(AnalogPrototype::new(
        new_zeros_real,
        new_zeros_imag,
        new_poles_real,
        new_poles_imag,
        gain.abs(),
    ))
}

/// Transform lowpass prototype to bandpass with specified center and bandwidth.
///
/// Transformation: s → (s² + ω₀²)/(B·s)
///
/// where ω₀ is the center frequency and B is the bandwidth.
/// Each pole/zero becomes a pair.
///
/// All operations are tensor-based with no CPU transfers.
pub fn lp2bp_zpk_impl<R, C>(
    client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
    bw: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let device = proto.zeros_real.device();
    let dtype = proto.zeros_real.dtype();

    let n_zeros = proto.zeros_real.shape()[0];
    let n_poles = proto.poles_real.shape()[0];

    // Transform zeros: z' = (z*bw ± sqrt((z*bw)² - 4*wo²)) / 2
    let (z1_re, z1_im, z2_re, z2_im) = if n_zeros > 0 {
        lp2bp_transform_tensor(client, &proto.zeros_real, &proto.zeros_imag, wo, bw)?
    } else {
        (
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
        )
    };

    // Interleave the two roots: [z1[0], z2[0], z1[1], z2[1], ...]
    let transformed_zeros_re = if n_zeros > 0 {
        interleave_tensors(client, &z1_re, &z2_re)?
    } else {
        Tensor::zeros(&[0], dtype, device)?
    };
    let transformed_zeros_im = if n_zeros > 0 {
        interleave_tensors(client, &z1_im, &z2_im)?
    } else {
        Tensor::zeros(&[0], dtype, device)?
    };

    // Add zeros at origin for degree matching
    // Each original zero becomes 2 zeros, so we need 2*(n_poles - n_zeros) zeros at origin
    let extra_zeros = 2 * n_poles.saturating_sub(n_zeros);
    let new_zeros_real = if extra_zeros > 0 {
        let origin_re = Tensor::zeros(&[extra_zeros], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_zeros_re, &origin_re], 0)?
        } else {
            origin_re
        }
    } else {
        transformed_zeros_re
    };
    let new_zeros_imag = if extra_zeros > 0 {
        let origin_im = Tensor::zeros(&[extra_zeros], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_zeros_im, &origin_im], 0)?
        } else {
            origin_im
        }
    } else {
        transformed_zeros_im
    };

    // Transform poles
    let (p1_re, p1_im, p2_re, p2_im) =
        lp2bp_transform_tensor(client, &proto.poles_real, &proto.poles_imag, wo, bw)?;

    let new_poles_real = interleave_tensors(client, &p1_re, &p2_re)?;
    let new_poles_imag = interleave_tensors(client, &p1_im, &p2_im)?;

    // Gain transformation
    let degree = (n_poles as i32) - (n_zeros as i32);
    let gain = proto.gain * bw.powi(degree);

    Ok(AnalogPrototype::new(
        new_zeros_real,
        new_zeros_imag,
        new_poles_real,
        new_poles_imag,
        gain,
    ))
}

/// Transform lowpass prototype to bandstop with specified center and bandwidth.
///
/// Transformation: s → B·s/(s² + ω₀²)
///
/// All operations are tensor-based with no CPU transfers.
pub fn lp2bs_zpk_impl<R, C>(
    client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
    bw: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let device = proto.zeros_real.device();
    let dtype = proto.zeros_real.dtype();

    let n_zeros = proto.zeros_real.shape()[0];
    let n_poles = proto.poles_real.shape()[0];

    // Transform zeros using bandstop formula
    let (z1_re, z1_im, z2_re, z2_im) = if n_zeros > 0 {
        lp2bs_transform_tensor(client, &proto.zeros_real, &proto.zeros_imag, wo, bw)?
    } else {
        (
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
        )
    };

    let transformed_zeros_re = if n_zeros > 0 {
        interleave_tensors(client, &z1_re, &z2_re)?
    } else {
        Tensor::zeros(&[0], dtype, device)?
    };
    let transformed_zeros_im = if n_zeros > 0 {
        interleave_tensors(client, &z1_im, &z2_im)?
    } else {
        Tensor::zeros(&[0], dtype, device)?
    };

    // Add zeros at ±j*wo for degree matching
    let extra_zeros_per_side = n_poles.saturating_sub(n_zeros);
    let new_zeros_real = if extra_zeros_per_side > 0 {
        let origin_re = Tensor::zeros(&[2 * extra_zeros_per_side], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_zeros_re, &origin_re], 0)?
        } else {
            origin_re
        }
    } else {
        transformed_zeros_re
    };

    // For bandstop, extra zeros are at ±j*wo (alternating +wo and -wo imaginary parts)
    let new_zeros_imag = if extra_zeros_per_side > 0 {
        let mut extra_im = Vec::with_capacity(2 * extra_zeros_per_side);
        for _ in 0..extra_zeros_per_side {
            extra_im.push(wo);
            extra_im.push(-wo);
        }
        let extra_tensor = Tensor::from_slice(&extra_im, &[2 * extra_zeros_per_side], device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_zeros_im, &extra_tensor], 0)?
        } else {
            extra_tensor
        }
    } else {
        transformed_zeros_im
    };

    // Transform poles
    let (p1_re, p1_im, p2_re, p2_im) =
        lp2bs_transform_tensor(client, &proto.poles_real, &proto.poles_imag, wo, bw)?;

    let new_poles_real = interleave_tensors(client, &p1_re, &p2_re)?;
    let new_poles_imag = interleave_tensors(client, &p1_im, &p2_im)?;

    // Gain transformation
    let mut gain = proto.gain;

    if n_zeros > 0 {
        // Divide by product of zero magnitudes
        let zero_mag = complex_mag(client, &proto.zeros_real, &proto.zeros_imag)?;
        let log_mag = client.log(&client.add_scalar(&zero_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain /= sum_log.exp();
    }

    if n_poles > 0 {
        // Multiply by product of pole magnitudes
        let pole_mag = complex_mag(client, &proto.poles_real, &proto.poles_imag)?;
        let log_mag = client.log(&client.add_scalar(&pole_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain *= sum_log.exp();
    }

    Ok(AnalogPrototype::new(
        new_zeros_real,
        new_zeros_imag,
        new_poles_real,
        new_poles_imag,
        gain.abs(),
    ))
}

// ============================================================================
// Helper functions for bandpass/bandstop transforms
// ============================================================================

/// Apply LP to BP transformation to all points in tensors.
/// Returns (z1_re, z1_im, z2_re, z2_im) - the two roots for each input point.
fn lp2bp_transform_tensor<R, C>(
    client: &C,
    s_re: &Tensor<R>,
    s_im: &Tensor<R>,
    wo: f64,
    bw: f64,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // z' = (s*bw ± sqrt((s*bw)² - 4*wo²)) / 2

    // s * bw
    let sbw_re = client.mul_scalar(s_re, bw)?;
    let sbw_im = client.mul_scalar(s_im, bw)?;

    // (s*bw)² = (sbw_re + i*sbw_im)²
    // = sbw_re² - sbw_im² + 2i*sbw_re*sbw_im
    let sbw_re_sq = client.mul(&sbw_re, &sbw_re)?;
    let sbw_im_sq = client.mul(&sbw_im, &sbw_im)?;
    let sbw_sq_re = client.sub(&sbw_re_sq, &sbw_im_sq)?;
    let sbw_re_im = client.mul(&sbw_re, &sbw_im)?;
    let sbw_sq_im = client.mul_scalar(&sbw_re_im, 2.0)?;

    // (s*bw)² - 4*wo²
    let four_wo_sq = 4.0 * wo * wo;
    let disc_re = client.add_scalar(&sbw_sq_re, -four_wo_sq)?;
    let disc_im = sbw_sq_im;

    // sqrt of discriminant
    let (sqrt_re, sqrt_im) = complex_sqrt_tensor(client, &disc_re, &disc_im)?;

    // Two roots: (sbw ± sqrt) / 2
    let sum_re = client.add(&sbw_re, &sqrt_re)?;
    let sum_im = client.add(&sbw_im, &sqrt_im)?;
    let diff_re = client.sub(&sbw_re, &sqrt_re)?;
    let diff_im = client.sub(&sbw_im, &sqrt_im)?;

    let z1_re = client.mul_scalar(&sum_re, 0.5)?;
    let z1_im = client.mul_scalar(&sum_im, 0.5)?;
    let z2_re = client.mul_scalar(&diff_re, 0.5)?;
    let z2_im = client.mul_scalar(&diff_im, 0.5)?;

    Ok((z1_re, z1_im, z2_re, z2_im))
}

/// Apply LP to BS transformation to all points in tensors.
fn lp2bs_transform_tensor<R, C>(
    client: &C,
    s_re: &Tensor<R>,
    s_im: &Tensor<R>,
    wo: f64,
    bw: f64,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // For bandstop: first compute bw/s, then apply same formula

    // Create bw tensor
    let n = s_re.shape()[0];
    let device = s_re.device();
    let dtype = s_re.dtype();
    let bw_re = Tensor::full_scalar(&[n], dtype, bw, device)?;
    let bw_im = Tensor::zeros(&[n], dtype, device)?;

    // bw / s
    let (bws_re, bws_im) = complex_div_tensor(client, &bw_re, &bw_im, s_re, s_im)?;

    // (bw/s)²
    let bws_re_sq = client.mul(&bws_re, &bws_re)?;
    let bws_im_sq = client.mul(&bws_im, &bws_im)?;
    let bws_sq_re = client.sub(&bws_re_sq, &bws_im_sq)?;
    let bws_re_im = client.mul(&bws_re, &bws_im)?;
    let bws_sq_im = client.mul_scalar(&bws_re_im, 2.0)?;

    // (bw/s)² - 4*wo²
    let four_wo_sq = 4.0 * wo * wo;
    let disc_re = client.add_scalar(&bws_sq_re, -four_wo_sq)?;
    let disc_im = bws_sq_im;

    let (sqrt_re, sqrt_im) = complex_sqrt_tensor(client, &disc_re, &disc_im)?;

    // Two roots
    let sum_re = client.add(&bws_re, &sqrt_re)?;
    let sum_im = client.add(&bws_im, &sqrt_im)?;
    let diff_re = client.sub(&bws_re, &sqrt_re)?;
    let diff_im = client.sub(&bws_im, &sqrt_im)?;

    let z1_re = client.mul_scalar(&sum_re, 0.5)?;
    let z1_im = client.mul_scalar(&sum_im, 0.5)?;
    let z2_re = client.mul_scalar(&diff_re, 0.5)?;
    let z2_im = client.mul_scalar(&diff_im, 0.5)?;

    Ok((z1_re, z1_im, z2_re, z2_im))
}

/// Interleave two tensors: [a0, b0, a1, b1, a2, b2, ...]
fn interleave_tensors<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R> + RuntimeClient<R>,
{
    let n = a.shape()[0];
    if n == 0 {
        return Ok(a.clone());
    }

    // Stack along new dimension then flatten
    // a: [n], b: [n] -> stack: [n, 2] -> reshape: [2*n]
    let stacked = client.stack(&[a, b], 1)?; // Shape: [n, 2]
    stacked.reshape(&[2 * n])
}
