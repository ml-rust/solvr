//! Frequency transformations for analog filter prototypes.
//!
//! These functions transform a lowpass analog prototype (cutoff = 1 rad/s)
//! to other filter types (highpass, bandpass, bandstop) with specified cutoffs.

// Allow indexed loops for vector operations that update in place
#![allow(clippy::needless_range_loop)]

use crate::signal::filter::types::AnalogPrototype;
use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Transform lowpass prototype to lowpass with specified cutoff.
///
/// Transformation: s → s/ω₀
pub fn lp2lp_zpk_impl<R, C>(
    _client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let device = proto.zeros_real.device();

    // Transform zeros: z → z * ω₀
    let z_re: Vec<f64> = proto.zeros_real.to_vec();
    let z_im: Vec<f64> = proto.zeros_imag.to_vec();
    let p_re: Vec<f64> = proto.poles_real.to_vec();
    let p_im: Vec<f64> = proto.poles_imag.to_vec();

    let new_z_re: Vec<f64> = z_re.iter().map(|&x| x * wo).collect();
    let new_z_im: Vec<f64> = z_im.iter().map(|&x| x * wo).collect();
    let new_p_re: Vec<f64> = p_re.iter().map(|&x| x * wo).collect();
    let new_p_im: Vec<f64> = p_im.iter().map(|&x| x * wo).collect();

    // Gain transformation
    let degree = (p_re.len() as i32) - (z_re.len() as i32);
    let gain = proto.gain * wo.powi(degree);

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&new_z_re, &[new_z_re.len()], device),
        Tensor::from_slice(&new_z_im, &[new_z_im.len()], device),
        Tensor::from_slice(&new_p_re, &[new_p_re.len()], device),
        Tensor::from_slice(&new_p_im, &[new_p_im.len()], device),
        gain,
    ))
}

/// Transform lowpass prototype to highpass with specified cutoff.
///
/// Transformation: s → ω₀/s
///
/// This inverts zeros and poles around the imaginary axis.
pub fn lp2hp_zpk_impl<R, C>(
    _client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let device = proto.zeros_real.device();
    let _dtype = proto.zeros_real.dtype();

    let z_re: Vec<f64> = proto.zeros_real.to_vec();
    let z_im: Vec<f64> = proto.zeros_imag.to_vec();
    let p_re: Vec<f64> = proto.poles_real.to_vec();
    let p_im: Vec<f64> = proto.poles_imag.to_vec();

    let n_zeros = z_re.len();
    let n_poles = p_re.len();

    // Transform existing zeros: z → ω₀/z
    let mut new_z_re = Vec::with_capacity(n_poles);
    let mut new_z_im = Vec::with_capacity(n_poles);

    for i in 0..n_zeros {
        let (re, im) = complex_divide(wo, 0.0, z_re[i], z_im[i]);
        new_z_re.push(re);
        new_z_im.push(im);
    }

    // Add zeros at origin for the degree difference
    for _ in n_zeros..n_poles {
        new_z_re.push(0.0);
        new_z_im.push(0.0);
    }

    // Transform poles: p → ω₀/p
    let mut new_p_re = Vec::with_capacity(n_poles);
    let mut new_p_im = Vec::with_capacity(n_poles);

    for i in 0..n_poles {
        let (re, im) = complex_divide(wo, 0.0, p_re[i], p_im[i]);
        new_p_re.push(re);
        new_p_im.push(im);
    }

    // Gain transformation
    // k' = k * prod(-z_i) / prod(-p_i)
    let mut gain = proto.gain;

    for i in 0..n_zeros {
        let mag = (z_re[i] * z_re[i] + z_im[i] * z_im[i]).sqrt();
        gain *= mag;
    }

    for i in 0..n_poles {
        let mag = (p_re[i] * p_re[i] + p_im[i] * p_im[i]).sqrt();
        gain /= mag;
    }

    // Adjust sign based on real parts
    let mut sign = 1.0;
    for i in 0..n_zeros {
        if z_re[i] < 0.0 {
            sign = -sign;
        }
    }
    for i in 0..n_poles {
        if p_re[i] < 0.0 {
            sign = -sign;
        }
    }
    gain *= sign;

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&new_z_re, &[new_z_re.len()], device),
        Tensor::from_slice(&new_z_im, &[new_z_im.len()], device),
        Tensor::from_slice(&new_p_re, &[new_p_re.len()], device),
        Tensor::from_slice(&new_p_im, &[new_p_im.len()], device),
        gain.abs(),
    ))
}

/// Transform lowpass prototype to bandpass with specified center and bandwidth.
///
/// Transformation: s → (s² + ω₀²)/(B·s)
///
/// where ω₀ is the center frequency and B is the bandwidth.
/// Each pole/zero becomes a pair.
pub fn lp2bp_zpk_impl<R, C>(
    _client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
    bw: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let device = proto.zeros_real.device();

    let z_re: Vec<f64> = proto.zeros_real.to_vec();
    let z_im: Vec<f64> = proto.zeros_imag.to_vec();
    let p_re: Vec<f64> = proto.poles_real.to_vec();
    let p_im: Vec<f64> = proto.poles_imag.to_vec();

    let n_zeros = z_re.len();
    let n_poles = p_re.len();

    // Each zero/pole becomes two
    // z' = (z*bw ± sqrt((z*bw)² - 4*wo²)) / 2
    let mut new_z_re = Vec::with_capacity(2 * n_poles);
    let mut new_z_im = Vec::with_capacity(2 * n_poles);

    for i in 0..n_zeros {
        let (z1_re, z1_im, z2_re, z2_im) = lp2bp_transform_point(z_re[i], z_im[i], wo, bw);
        new_z_re.push(z1_re);
        new_z_im.push(z1_im);
        new_z_re.push(z2_re);
        new_z_im.push(z2_im);
    }

    // Add zeros at origin for degree matching
    for _ in n_zeros..n_poles {
        new_z_re.push(0.0);
        new_z_im.push(0.0);
        new_z_re.push(0.0);
        new_z_im.push(0.0);
    }

    let mut new_p_re = Vec::with_capacity(2 * n_poles);
    let mut new_p_im = Vec::with_capacity(2 * n_poles);

    for i in 0..n_poles {
        let (p1_re, p1_im, p2_re, p2_im) = lp2bp_transform_point(p_re[i], p_im[i], wo, bw);
        new_p_re.push(p1_re);
        new_p_im.push(p1_im);
        new_p_re.push(p2_re);
        new_p_im.push(p2_im);
    }

    // Gain transformation
    let degree = (n_poles as i32) - (n_zeros as i32);
    let gain = proto.gain * bw.powi(degree);

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&new_z_re, &[new_z_re.len()], device),
        Tensor::from_slice(&new_z_im, &[new_z_im.len()], device),
        Tensor::from_slice(&new_p_re, &[new_p_re.len()], device),
        Tensor::from_slice(&new_p_im, &[new_p_im.len()], device),
        gain,
    ))
}

/// Transform lowpass prototype to bandstop with specified center and bandwidth.
///
/// Transformation: s → B·s/(s² + ω₀²)
pub fn lp2bs_zpk_impl<R, C>(
    _client: &C,
    proto: &AnalogPrototype<R>,
    wo: f64,
    bw: f64,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let device = proto.zeros_real.device();

    let z_re: Vec<f64> = proto.zeros_real.to_vec();
    let z_im: Vec<f64> = proto.zeros_imag.to_vec();
    let p_re: Vec<f64> = proto.poles_real.to_vec();
    let p_im: Vec<f64> = proto.poles_imag.to_vec();

    let n_zeros = z_re.len();
    let n_poles = p_re.len();

    // Transform zeros
    let mut new_z_re = Vec::with_capacity(2 * n_poles);
    let mut new_z_im = Vec::with_capacity(2 * n_poles);

    for i in 0..n_zeros {
        let (z1_re, z1_im, z2_re, z2_im) = lp2bs_transform_point(z_re[i], z_im[i], wo, bw);
        new_z_re.push(z1_re);
        new_z_im.push(z1_im);
        new_z_re.push(z2_re);
        new_z_im.push(z2_im);
    }

    // Add zeros at ±j*wo for degree matching
    for _ in n_zeros..n_poles {
        new_z_re.push(0.0);
        new_z_im.push(wo);
        new_z_re.push(0.0);
        new_z_im.push(-wo);
    }

    // Transform poles
    let mut new_p_re = Vec::with_capacity(2 * n_poles);
    let mut new_p_im = Vec::with_capacity(2 * n_poles);

    for i in 0..n_poles {
        let (p1_re, p1_im, p2_re, p2_im) = lp2bs_transform_point(p_re[i], p_im[i], wo, bw);
        new_p_re.push(p1_re);
        new_p_im.push(p1_im);
        new_p_re.push(p2_re);
        new_p_im.push(p2_im);
    }

    // Gain transformation
    let mut gain = proto.gain;

    for i in 0..n_zeros {
        let mag = (z_re[i] * z_re[i] + z_im[i] * z_im[i]).sqrt();
        gain /= mag;
    }

    for i in 0..n_poles {
        let mag = (p_re[i] * p_re[i] + p_im[i] * p_im[i]).sqrt();
        gain *= mag;
    }

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&new_z_re, &[new_z_re.len()], device),
        Tensor::from_slice(&new_z_im, &[new_z_im.len()], device),
        Tensor::from_slice(&new_p_re, &[new_p_re.len()], device),
        Tensor::from_slice(&new_p_im, &[new_p_im.len()], device),
        gain.abs(),
    ))
}

// ============================================================================
// Helper functions
// ============================================================================

/// Complex division: (a + bi) / (c + di)
fn complex_divide(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let denom = c * c + d * d;
    if denom < 1e-30 {
        return (f64::INFINITY, 0.0);
    }
    let re = (a * c + b * d) / denom;
    let im = (b * c - a * d) / denom;
    (re, im)
}

/// Complex square root
fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
    let mag = (re * re + im * im).sqrt();
    let sqrt_mag = mag.sqrt();

    if sqrt_mag < 1e-30 {
        return (0.0, 0.0);
    }

    let angle = im.atan2(re);
    let half_angle = angle / 2.0;

    (sqrt_mag * half_angle.cos(), sqrt_mag * half_angle.sin())
}

/// Transform a point for LP to BP transformation.
///
/// Returns two points: (z1_re, z1_im, z2_re, z2_im)
fn lp2bp_transform_point(s_re: f64, s_im: f64, wo: f64, bw: f64) -> (f64, f64, f64, f64) {
    // z' = (s*bw ± sqrt((s*bw)² - 4*wo²)) / 2

    // s * bw
    let sbw_re = s_re * bw;
    let sbw_im = s_im * bw;

    // (s*bw)²
    let sbw_sq_re = sbw_re * sbw_re - sbw_im * sbw_im;
    let sbw_sq_im = 2.0 * sbw_re * sbw_im;

    // (s*bw)² - 4*wo²
    let disc_re = sbw_sq_re - 4.0 * wo * wo;
    let disc_im = sbw_sq_im;

    // sqrt of discriminant
    let (sqrt_re, sqrt_im) = complex_sqrt(disc_re, disc_im);

    // Two roots
    let z1_re = (sbw_re + sqrt_re) / 2.0;
    let z1_im = (sbw_im + sqrt_im) / 2.0;
    let z2_re = (sbw_re - sqrt_re) / 2.0;
    let z2_im = (sbw_im - sqrt_im) / 2.0;

    (z1_re, z1_im, z2_re, z2_im)
}

/// Transform a point for LP to BS transformation.
fn lp2bs_transform_point(s_re: f64, s_im: f64, wo: f64, bw: f64) -> (f64, f64, f64, f64) {
    // For bandstop: first compute bw/s, then apply same formula

    // bw / s
    let (bws_re, bws_im) = complex_divide(bw, 0.0, s_re, s_im);

    // (bw/s)²
    let bws_sq_re = bws_re * bws_re - bws_im * bws_im;
    let bws_sq_im = 2.0 * bws_re * bws_im;

    // (bw/s)² - 4*wo²
    let disc_re = bws_sq_re - 4.0 * wo * wo;
    let disc_im = bws_sq_im;

    let (sqrt_re, sqrt_im) = complex_sqrt(disc_re, disc_im);

    let z1_re = (bws_re + sqrt_re) / 2.0;
    let z1_im = (bws_im + sqrt_im) / 2.0;
    let z2_re = (bws_re - sqrt_re) / 2.0;
    let z2_im = (bws_im - sqrt_im) / 2.0;

    (z1_re, z1_im, z2_re, z2_im)
}
