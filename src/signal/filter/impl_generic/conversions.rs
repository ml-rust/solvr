//! Generic implementations of filter representation conversions.
//!
//! Converts between transfer function (b, a), zero-pole-gain (zpk),
//! and second-order sections (sos) representations.

use crate::signal::filter::traits::conversions::SosPairing;
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Complex number represented as (real, imaginary) tuple.
type Complex = (f64, f64);

/// Pair of complex numbers (for pole/zero pairing in biquad sections).
type ComplexPair = (Complex, Complex);

/// Convert transfer function to zeros, poles, and gain.
///
/// Uses polynomial root finding via companion matrix eigendecomposition.
pub fn tf2zpk_impl<R, C>(client: &C, tf: &TransferFunction<R>) -> Result<ZpkFilter<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let b = &tf.b;
    let a = &tf.a;

    // Validate inputs
    if b.ndim() != 1 || a.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tf",
            reason: "Transfer function coefficients must be 1D".to_string(),
        });
    }

    if b.shape()[0] == 0 || a.shape()[0] == 0 {
        return Err(Error::InvalidArgument {
            arg: "tf",
            reason: "Transfer function coefficients cannot be empty".to_string(),
        });
    }

    // Coefficients are in descending order, need to reverse for polyroots
    // which expects ascending order
    let b_ascending = b.flip(0)?.contiguous();
    let a_ascending = a.flip(0)?.contiguous();

    // Find zeros (roots of numerator)
    let zeros = if b.shape()[0] > 1 {
        client.polyroots(&b_ascending)?
    } else {
        // Constant numerator has no zeros
        let device = b.device();
        numr::algorithm::polynomial::types::PolynomialRoots {
            roots_real: Tensor::zeros(&[0], b.dtype(), device),
            roots_imag: Tensor::zeros(&[0], b.dtype(), device),
        }
    };

    // Find poles (roots of denominator)
    let poles = if a.shape()[0] > 1 {
        client.polyroots(&a_ascending)?
    } else {
        let device = a.device();
        numr::algorithm::polynomial::types::PolynomialRoots {
            roots_real: Tensor::zeros(&[0], a.dtype(), device),
            roots_imag: Tensor::zeros(&[0], a.dtype(), device),
        }
    };

    // Compute gain: ratio of leading coefficients (highest power = index 0)
    let b0: f64 = b.narrow(0, 0, 1)?.to_vec()[0];
    let a0: f64 = a.narrow(0, 0, 1)?.to_vec()[0];
    let gain = b0 / a0;

    Ok(ZpkFilter::new(
        zeros.roots_real,
        zeros.roots_imag,
        poles.roots_real,
        poles.roots_imag,
        gain,
    ))
}

/// Convert zeros, poles, and gain to transfer function.
///
/// Uses polynomial multiplication via convolution.
pub fn zpk2tf_impl<R, C>(client: &C, zpk: &ZpkFilter<R>) -> Result<TransferFunction<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = zpk.zeros_real.device();
    let _dtype = zpk.zeros_real.dtype();

    // Build numerator from zeros
    let b = if zpk.num_zeros() == 0 {
        // No zeros: numerator is just [gain]
        Tensor::from_slice(&[zpk.gain], &[1], device)
    } else {
        // Build polynomial from roots, then scale by gain
        let b_monic = client.polyfromroots(&zpk.zeros_real, &zpk.zeros_imag)?;
        // polyfromroots returns ascending order, we need descending
        let b_desc = b_monic.flip(0)?.contiguous();
        client.mul_scalar(&b_desc, zpk.gain)?
    };

    // Build denominator from poles
    let a = if zpk.num_poles() == 0 {
        // No poles: denominator is just [1]
        Tensor::from_slice(&[1.0], &[1], device)
    } else {
        let a_monic = client.polyfromroots(&zpk.poles_real, &zpk.poles_imag)?;
        // polyfromroots returns ascending order, we need descending
        a_monic.flip(0)?.contiguous()
    };

    Ok(TransferFunction::new(b, a))
}

/// Convert transfer function to second-order sections.
pub fn tf2sos_impl<R, C>(
    client: &C,
    tf: &TransferFunction<R>,
    pairing: Option<SosPairing>,
) -> Result<SosFilter<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let zpk = tf2zpk_impl(client, tf)?;
    zpk2sos_impl(client, &zpk, pairing)
}

/// Convert second-order sections to transfer function.
pub fn sos2tf_impl<R, C>(client: &C, sos: &SosFilter<R>) -> Result<TransferFunction<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n_sections = sos.num_sections();
    let device = sos.sections.device();
    let _dtype = sos.sections.dtype();

    if n_sections == 0 {
        // Empty filter
        return Ok(TransferFunction::new(
            Tensor::from_slice(&[1.0], &[1], device),
            Tensor::from_slice(&[1.0], &[1], device),
        ));
    }

    // Start with first section
    // Each section is [b0, b1, b2, a0, a1, a2]
    let first_section = sos.sections.narrow(0, 0, 1)?.reshape(&[6])?;
    let section_data: Vec<f64> = first_section.to_vec();

    let mut b = Tensor::from_slice(&section_data[0..3], &[3], device);
    let mut a = Tensor::from_slice(&section_data[3..6], &[3], device);

    // Multiply by remaining sections
    for i in 1..n_sections {
        let section = sos.sections.narrow(0, i, 1)?.reshape(&[6])?;
        let section_data: Vec<f64> = section.to_vec();

        let bi = Tensor::from_slice(&section_data[0..3], &[3], device);
        let ai = Tensor::from_slice(&section_data[3..6], &[3], device);

        // Flip to ascending order for polymul, then flip back
        let b_asc = b.flip(0)?.contiguous();
        let bi_asc = bi.flip(0)?.contiguous();
        let b_new_asc = client.polymul(&b_asc, &bi_asc)?;
        b = b_new_asc.flip(0)?.contiguous();

        let a_asc = a.flip(0)?.contiguous();
        let ai_asc = ai.flip(0)?.contiguous();
        let a_new_asc = client.polymul(&a_asc, &ai_asc)?;
        a = a_new_asc.flip(0)?.contiguous();
    }

    Ok(TransferFunction::new(b, a))
}

/// Convert zeros, poles, gain to second-order sections.
///
/// This is the core conversion that pairs poles/zeros into biquad sections.
pub fn zpk2sos_impl<R, C>(
    _client: &C,
    zpk: &ZpkFilter<R>,
    pairing: Option<SosPairing>,
) -> Result<SosFilter<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let _pairing = pairing.unwrap_or_default();
    let device = zpk.zeros_real.device();
    let _dtype = zpk.zeros_real.dtype();

    let n_zeros = zpk.num_zeros();
    let n_poles = zpk.num_poles();

    // Number of sections needed
    let n_sections = n_poles.max(n_zeros).div_ceil(2);

    if n_sections == 0 {
        // Just a gain
        let sections = Tensor::from_slice(&[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 6], device);
        return Ok(SosFilter::new(sections));
    }

    // Get poles and zeros as vectors for pairing
    let zeros_re: Vec<f64> = zpk.zeros_real.to_vec();
    let zeros_im: Vec<f64> = zpk.zeros_imag.to_vec();
    let poles_re: Vec<f64> = zpk.poles_real.to_vec();
    let poles_im: Vec<f64> = zpk.poles_imag.to_vec();

    // Sort and pair poles/zeros
    // Complex conjugate pairs go together, then pair with nearest
    let (paired_zeros, paired_poles) =
        pair_poles_zeros(&zeros_re, &zeros_im, &poles_re, &poles_im)?;

    // Build sections
    let mut sections_data = Vec::with_capacity(n_sections * 6);
    let remaining_gain = zpk.gain;

    for i in 0..n_sections {
        let (b0, b1, b2, a0, a1, a2) = if i < paired_poles.len() {
            let (p1, p2) = paired_poles[i];
            let (z1, z2) = if i < paired_zeros.len() {
                paired_zeros[i]
            } else {
                // No more zeros, use (0, 0) which adds z^-2 term
                ((0.0, 0.0), (0.0, 0.0))
            };

            biquad_coeffs(z1, z2, p1, p2)
        } else {
            // Extra zeros without poles (shouldn't happen for proper filters)
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        };

        // Distribute gain across sections (put it all in first section)
        let scale = if i == 0 { remaining_gain } else { 1.0 };

        sections_data.extend_from_slice(&[b0 * scale, b1 * scale, b2 * scale, a0, a1, a2]);
    }

    let sections = Tensor::from_slice(&sections_data, &[n_sections, 6], device);
    Ok(SosFilter::new(sections))
}

/// Convert second-order sections to zeros, poles, gain.
pub fn sos2zpk_impl<R, C>(_client: &C, sos: &SosFilter<R>) -> Result<ZpkFilter<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + RuntimeClient<R>,
{
    let n_sections = sos.num_sections();
    let device = sos.sections.device();
    let dtype = sos.sections.dtype();

    if n_sections == 0 {
        return Ok(ZpkFilter::new(
            Tensor::zeros(&[0], dtype, device),
            Tensor::zeros(&[0], dtype, device),
            Tensor::zeros(&[0], dtype, device),
            Tensor::zeros(&[0], dtype, device),
            1.0,
        ));
    }

    let mut all_zeros_re = Vec::new();
    let mut all_zeros_im = Vec::new();
    let mut all_poles_re = Vec::new();
    let mut all_poles_im = Vec::new();
    let mut gain = 1.0;

    for i in 0..n_sections {
        let section = sos.sections.narrow(0, i, 1)?.reshape(&[6])?;
        let coeffs: Vec<f64> = section.to_vec();

        let (b0, b1, b2) = (coeffs[0], coeffs[1], coeffs[2]);
        let (a0, a1, a2) = (coeffs[3], coeffs[4], coeffs[5]);

        // Accumulate gain from numerator leading coefficient
        gain *= b0 / a0;

        // Find zeros of numerator (roots of b0 + b1*z^-1 + b2*z^-2)
        // = roots of b2*z^2 + b1*z + b0 (after multiplying by z^2)
        let (z1_re, z1_im, z2_re, z2_im) = quadratic_roots(b2, b1, b0);
        if b2.abs() > 1e-14 || b1.abs() > 1e-14 {
            all_zeros_re.push(z1_re);
            all_zeros_im.push(z1_im);
            if b2.abs() > 1e-14 {
                all_zeros_re.push(z2_re);
                all_zeros_im.push(z2_im);
            }
        }

        // Find poles of denominator
        let (p1_re, p1_im, p2_re, p2_im) = quadratic_roots(a2, a1, a0);
        if a2.abs() > 1e-14 || a1.abs() > 1e-14 {
            all_poles_re.push(p1_re);
            all_poles_im.push(p1_im);
            if a2.abs() > 1e-14 {
                all_poles_re.push(p2_re);
                all_poles_im.push(p2_im);
            }
        }
    }

    let n_zeros = all_zeros_re.len();
    let n_poles = all_poles_re.len();

    Ok(ZpkFilter::new(
        Tensor::from_slice(&all_zeros_re, &[n_zeros], device),
        Tensor::from_slice(&all_zeros_im, &[n_zeros], device),
        Tensor::from_slice(&all_poles_re, &[n_poles], device),
        Tensor::from_slice(&all_poles_im, &[n_poles], device),
        gain,
    ))
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find roots of quadratic az^2 + bz + c.
fn quadratic_roots(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    if a.abs() < 1e-14 {
        // Linear equation
        if b.abs() < 1e-14 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let root = -c / b;
        return (root, 0.0, 0.0, 0.0);
    }

    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        // Real roots
        let sqrt_disc = disc.sqrt();
        let r1 = (-b + sqrt_disc) / (2.0 * a);
        let r2 = (-b - sqrt_disc) / (2.0 * a);
        (r1, 0.0, r2, 0.0)
    } else {
        // Complex conjugate roots
        let real = -b / (2.0 * a);
        let imag = (-disc).sqrt() / (2.0 * a);
        (real, imag, real, -imag)
    }
}

/// Pair poles and zeros for SOS conversion.
///
/// Returns pairs of (zero, zero) and (pole, pole) for each section.
fn pair_poles_zeros(
    zeros_re: &[f64],
    zeros_im: &[f64],
    poles_re: &[f64],
    poles_im: &[f64],
) -> Result<(Vec<ComplexPair>, Vec<ComplexPair>)> {
    // Separate into complex conjugate pairs and real values
    let mut complex_poles: Vec<(f64, f64)> = Vec::new();
    let mut real_poles: Vec<f64> = Vec::new();

    let mut i = 0;
    while i < poles_re.len() {
        if poles_im[i].abs() > 1e-10 {
            // Complex pole - find its conjugate
            complex_poles.push((poles_re[i], poles_im[i].abs()));
            // Skip conjugate if it exists
            if i + 1 < poles_im.len() && (poles_im[i] + poles_im[i + 1]).abs() < 1e-10 {
                i += 1;
            }
        } else {
            real_poles.push(poles_re[i]);
        }
        i += 1;
    }

    let mut complex_zeros: Vec<(f64, f64)> = Vec::new();
    let mut real_zeros: Vec<f64> = Vec::new();

    i = 0;
    while i < zeros_re.len() {
        if zeros_im[i].abs() > 1e-10 {
            complex_zeros.push((zeros_re[i], zeros_im[i].abs()));
            if i + 1 < zeros_im.len() && (zeros_im[i] + zeros_im[i + 1]).abs() < 1e-10 {
                i += 1;
            }
        } else {
            real_zeros.push(zeros_re[i]);
        }
        i += 1;
    }

    // Build pole pairs (complex pairs first, then real pairs)
    let mut pole_pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();

    for (re, im) in &complex_poles {
        pole_pairs.push(((*re, *im), (*re, -*im)));
    }

    // Pair real poles
    let mut j = 0;
    while j + 1 < real_poles.len() {
        pole_pairs.push(((real_poles[j], 0.0), (real_poles[j + 1], 0.0)));
        j += 2;
    }
    if j < real_poles.len() {
        // Odd pole
        pole_pairs.push(((real_poles[j], 0.0), (0.0, 0.0)));
    }

    // Build zero pairs similarly
    let mut zero_pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();

    for (re, im) in &complex_zeros {
        zero_pairs.push(((*re, *im), (*re, -*im)));
    }

    j = 0;
    while j + 1 < real_zeros.len() {
        zero_pairs.push(((real_zeros[j], 0.0), (real_zeros[j + 1], 0.0)));
        j += 2;
    }
    if j < real_zeros.len() {
        zero_pairs.push(((real_zeros[j], 0.0), (0.0, 0.0)));
    }

    Ok((zero_pairs, pole_pairs))
}

/// Compute biquad coefficients from a zero pair and pole pair.
///
/// Returns (b0, b1, b2, a0, a1, a2) for:
/// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
fn biquad_coeffs(
    z1: (f64, f64),
    z2: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    // Numerator: (z - z1)(z - z2) = z^2 - (z1+z2)z + z1*z2
    // In z^-1 form: 1 - (z1+z2)z^-1 + z1*z2*z^-2
    let (z1_re, z1_im) = z1;
    let (z2_re, z2_im) = z2;

    let b0 = 1.0;
    let b1 = -(z1_re + z2_re);
    // z1*z2 = (z1_re + i*z1_im)(z2_re + i*z2_im)
    // Real part = z1_re*z2_re - z1_im*z2_im
    let b2 = z1_re * z2_re - z1_im * z2_im;

    // Denominator: (z - p1)(z - p2)
    let (p1_re, p1_im) = p1;
    let (p2_re, p2_im) = p2;

    let a0 = 1.0;
    let a1 = -(p1_re + p2_re);
    let a2 = p1_re * p2_re - p1_im * p2_im;

    (b0, b1, b2, a0, a1, a2)
}
