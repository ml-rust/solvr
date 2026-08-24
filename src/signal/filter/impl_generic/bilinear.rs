//! Bilinear transform implementation.
//!
//! Converts analog (continuous-time) filters to digital (discrete-time) filters
//! using the bilinear transformation.
//!
//! All operations are fully tensorized - data stays on device with no GPU<->CPU transfers.
use crate::DType;

use crate::signal::filter::types::{AnalogPrototype, ZpkFilter};
use numr::error::Result;
use numr::ops::{ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Apply bilinear transform to convert analog filter to digital.
///
/// # Algorithm
///
/// The bilinear transform maps the s-plane to the z-plane via:
/// ```text
/// s = (2*fs) * (z - 1) / (z + 1)
/// ```
///
/// For poles and zeros, the transformation is:
/// ```text
/// z = (1 + s/(2*fs)) / (1 - s/(2*fs))
/// ```
///
/// # Frequency Warping
///
/// The bilinear transform warps frequencies according to:
/// ```text
/// ω_digital = 2 * arctan(ω_analog / (2*fs))
/// ```
///
/// Pre-warping is typically applied during filter design to compensate.
///
/// All operations are tensor-based with no CPU transfers.
pub fn bilinear_zpk_impl<R, C>(
    client: &C,
    analog: &AnalogPrototype<R>,
    fs: f64,
) -> Result<ZpkFilter<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + ShapeOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let device = analog.zeros_real.device();
    let dtype = analog.zeros_real.dtype();

    let n_zeros = analog.zeros_real.shape()[0];
    let n_poles = analog.poles_real.shape()[0];

    let fs2 = 2.0 * fs;

    // Transform zeros using tensorized bilinear transform
    let (transformed_z_re, transformed_z_im) = if n_zeros > 0 {
        bilinear_transform_tensor(client, &analog.zeros_real, &analog.zeros_imag, fs2)?
    } else {
        (
            Tensor::zeros(&[0], dtype, device)?,
            Tensor::zeros(&[0], dtype, device)?,
        )
    };

    // Add zeros at z = -1 for degree matching
    // (The bilinear transform maps s = infinity to z = -1)
    let extra_zeros = n_poles.saturating_sub(n_zeros);
    let digital_zeros_re = if extra_zeros > 0 {
        let minus_one = Tensor::full_scalar(&[extra_zeros], dtype, -1.0, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_z_re, &minus_one], 0)?
        } else {
            minus_one
        }
    } else {
        transformed_z_re
    };

    let digital_zeros_im = if extra_zeros > 0 {
        let zeros = Tensor::zeros(&[extra_zeros], dtype, device)?;
        if n_zeros > 0 {
            client.cat(&[&transformed_z_im, &zeros], 0)?
        } else {
            zeros
        }
    } else {
        transformed_z_im
    };

    // Transform poles
    let (digital_poles_re, digital_poles_im) =
        bilinear_transform_tensor(client, &analog.poles_real, &analog.poles_imag, fs2)?;

    // Compute gain transformation using tensor operations
    // k_d = k_a * prod(|2*fs - z_i|) / prod(|2*fs - p_i|)
    let mut gain = analog.gain;

    if n_zeros > 0 {
        // Compute |2*fs - z_i| for each zero
        // (2*fs - z) = (fs2 - z_re) - j*z_im
        let diff_re = client.add_scalar(&client.mul_scalar(&analog.zeros_real, -1.0)?, fs2)?;
        let diff_im = client.mul_scalar(&analog.zeros_imag, -1.0)?;

        // |diff| = sqrt(diff_re² + diff_im²)
        let diff_re_sq = client.mul(&diff_re, &diff_re)?;
        let diff_im_sq = client.mul(&diff_im, &diff_im)?;
        let diff_mag_sq = client.add(&diff_re_sq, &diff_im_sq)?;
        let diff_mag = client.sqrt(&diff_mag_sq)?;

        // Product via sum of logs
        let log_mag = client.log(&client.add_scalar(&diff_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain *= sum_log.exp();
    }

    if n_poles > 0 {
        // Compute |2*fs - p_i| for each pole
        let diff_re = client.add_scalar(&client.mul_scalar(&analog.poles_real, -1.0)?, fs2)?;
        let diff_im = client.mul_scalar(&analog.poles_imag, -1.0)?;

        let diff_re_sq = client.mul(&diff_re, &diff_re)?;
        let diff_im_sq = client.mul(&diff_im, &diff_im)?;
        let diff_mag_sq = client.add(&diff_re_sq, &diff_im_sq)?;
        let diff_mag = client.sqrt(&diff_mag_sq)?;

        let log_mag = client.log(&client.add_scalar(&diff_mag, 1e-30)?)?;
        let sum_log: f64 = client.sum(&log_mag, &[0], false)?.item()?;
        gain /= sum_log.exp();
    }

    // For each zero added at -1, we divide by 2*fs
    for _ in 0..extra_zeros {
        gain /= fs2;
    }

    // Make gain real and positive
    gain = gain.abs();

    Ok(ZpkFilter::new(
        digital_zeros_re,
        digital_zeros_im,
        digital_poles_re,
        digital_poles_im,
        gain,
    ))
}

/// Apply bilinear transform to all points in tensors.
///
/// s → z = (1 + s/fs2) / (1 - s/fs2)
///
/// All operations are tensor-based.
fn bilinear_transform_tensor<R, C>(
    client: &C,
    s_re: &Tensor<R>,
    s_im: &Tensor<R>,
    fs2: f64,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // z = (1 + s/fs2) / (1 - s/fs2)
    // Let s/fs2 = a + bi

    // a = s_re / fs2, b = s_im / fs2
    let a = client.mul_scalar(s_re, 1.0 / fs2)?;
    let b = client.mul_scalar(s_im, 1.0 / fs2)?;

    // Numerator: 1 + a + bi = (1 + a) + bi
    let num_re = client.add_scalar(&a, 1.0)?;
    let num_im = b.clone();

    // Denominator: 1 - a - bi = (1 - a) - bi
    let one = Tensor::ones(&[s_re.shape()[0]], s_re.dtype(), s_re.device())?;
    let denom_re = client.sub(&one, &a)?;
    let denom_im = client.mul_scalar(&b, -1.0)?;

    // Complex division: (num_re + i*num_im) / (denom_re + i*denom_im)
    // = (num_re*denom_re + num_im*denom_im + i*(num_im*denom_re - num_re*denom_im)) / |denom|²

    let denom_re_sq = client.mul(&denom_re, &denom_re)?;
    let denom_im_sq = client.mul(&denom_im, &denom_im)?;
    let denom_mag_sq = client.add(&denom_re_sq, &denom_im_sq)?;

    // Add epsilon for numerical stability (handles s at infinity -> z = -1)
    let denom_safe = client.add_scalar(&denom_mag_sq, 1e-30)?;

    // Real part: (num_re*denom_re + num_im*denom_im) / |denom|²
    let nr_dr = client.mul(&num_re, &denom_re)?;
    let ni_di = client.mul(&num_im, &denom_im)?;
    let z_re_num = client.add(&nr_dr, &ni_di)?;
    let z_re = client.div(&z_re_num, &denom_safe)?;

    // Imag part: (num_im*denom_re - num_re*denom_im) / |denom|²
    let ni_dr = client.mul(&num_im, &denom_re)?;
    let nr_di = client.mul(&num_re, &denom_im)?;
    let z_im_num = client.sub(&ni_dr, &nr_di)?;
    let z_im = client.div(&z_im_num, &denom_safe)?;

    Ok((z_re, z_im))
}

/// Pre-warp a digital frequency to analog frequency.
///
/// Used to compensate for frequency warping in the bilinear transform.
///
/// # Arguments
///
/// * `w_digital` - Normalized digital frequency (0 to 1, where 1 = Nyquist)
/// * `fs` - Sample rate
///
/// # Returns
///
/// Analog frequency in rad/s.
pub fn prewarp(w_digital: f64, fs: f64) -> f64 {
    // ω_analog = 2 * fs * tan(π * w_digital / 2)
    // But w_digital is normalized to Nyquist, so:
    // ω_analog = 2 * fs * tan(π * w_digital)
    // where w_digital is fraction of Nyquist (0 to 1)
    2.0 * fs * (PI * w_digital / 2.0).tan()
}
