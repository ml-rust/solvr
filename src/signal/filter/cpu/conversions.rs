//! CPU implementation of filter conversions.
//!
//! Some conversions (sos2tf, zpk2sos, sos2zpk) are CPU-only because they involve
//! inherently sequential algorithms (pole/zero pairing, quadratic root finding)
//! with tiny data sizes (n_sections typically 1-10, each 6 floats) where GPU
//! transfer overhead far exceeds the computation time.

use crate::signal::filter::impl_generic::{tf2zpk_impl, zpk2tf_impl};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Complex number represented as (real, imaginary) tuple.
type Complex = (f64, f64);

/// Pair of complex numbers (for pole/zero pairing in biquad sections).
type ComplexPair = (Complex, Complex);

impl FilterConversions<CpuRuntime> for CpuClient {
    fn tf2zpk(&self, tf: &TransferFunction<CpuRuntime>) -> Result<ZpkFilter<CpuRuntime>> {
        tf2zpk_impl(self, tf)
    }

    fn zpk2tf(&self, zpk: &ZpkFilter<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
        zpk2tf_impl(self, zpk)
    }

    fn tf2sos(
        &self,
        tf: &TransferFunction<CpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CpuRuntime>> {
        let zpk = tf2zpk_impl(self, tf)?;
        self.zpk2sos(&zpk, pairing)
    }

    fn sos2tf(&self, sos: &SosFilter<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
        sos2tf_cpu(self, sos)
    }

    fn zpk2sos(
        &self,
        zpk: &ZpkFilter<CpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CpuRuntime>> {
        zpk2sos_cpu(zpk, pairing)
    }

    fn sos2zpk(&self, sos: &SosFilter<CpuRuntime>) -> Result<ZpkFilter<CpuRuntime>> {
        sos2zpk_cpu(sos)
    }
}

// ============================================================================
// CPU-only SOS conversion implementations
// ============================================================================

/// Convert second-order sections to transfer function (CPU-only).
///
/// This involves iterative polynomial multiplication which is inherently sequential
/// and operates on tiny data (n_sections * 6 coefficients).
fn sos2tf_cpu(
    client: &CpuClient,
    sos: &SosFilter<CpuRuntime>,
) -> Result<TransferFunction<CpuRuntime>> {
    use numr::algorithm::polynomial::PolynomialAlgorithms;

    let n_sections = sos.num_sections();
    let device = sos.sections.device();

    if n_sections == 0 {
        return Ok(TransferFunction::new(
            Tensor::from_slice(&[1.0], &[1], device),
            Tensor::from_slice(&[1.0], &[1], device),
        ));
    }

    // Get all section data at once (API boundary)
    let sections_data: Vec<f64> = sos.sections.to_vec();

    // Start with first section [b0, b1, b2, a0, a1, a2]
    let mut b = Tensor::from_slice(&sections_data[0..3], &[3], device);
    let mut a = Tensor::from_slice(&sections_data[3..6], &[3], device);

    // Multiply by remaining sections
    for i in 1..n_sections {
        let offset = i * 6;
        let bi = Tensor::from_slice(&sections_data[offset..offset + 3], &[3], device);
        let ai = Tensor::from_slice(&sections_data[offset + 3..offset + 6], &[3], device);

        // Flip to ascending order for polymul, then flip back
        let b_asc = b.flip(0)?.contiguous()?;
        let bi_asc = bi.flip(0)?.contiguous()?;
        let b_new_asc = client.polymul(&b_asc, &bi_asc)?;
        b = b_new_asc.flip(0)?.contiguous()?;

        let a_asc = a.flip(0)?.contiguous()?;
        let ai_asc = ai.flip(0)?.contiguous()?;
        let a_new_asc = client.polymul(&a_asc, &ai_asc)?;
        a = a_new_asc.flip(0)?.contiguous()?;
    }

    Ok(TransferFunction::new(b, a))
}

/// Convert zeros, poles, gain to second-order sections (CPU-only).
///
/// This is the core conversion that pairs poles/zeros into biquad sections.
/// The pairing algorithm is inherently sequential.
fn zpk2sos_cpu(
    zpk: &ZpkFilter<CpuRuntime>,
    pairing: Option<SosPairing>,
) -> Result<SosFilter<CpuRuntime>> {
    let _pairing = pairing.unwrap_or_default();
    let device = zpk.zeros_real.device();

    let n_zeros = zpk.num_zeros();
    let n_poles = zpk.num_poles();

    // Number of sections needed
    let n_sections = n_poles.max(n_zeros).div_ceil(2);

    if n_sections == 0 {
        // Just a gain
        let sections = Tensor::from_slice(&[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 6], device);
        return Ok(SosFilter::new(sections));
    }

    // Get poles and zeros as vectors (API boundary - small data)
    let zeros_re: Vec<f64> = zpk.zeros_real.to_vec();
    let zeros_im: Vec<f64> = zpk.zeros_imag.to_vec();
    let poles_re: Vec<f64> = zpk.poles_real.to_vec();
    let poles_im: Vec<f64> = zpk.poles_imag.to_vec();

    // Sort and pair poles/zeros
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
                ((0.0, 0.0), (0.0, 0.0))
            };

            biquad_coeffs(z1, z2, p1, p2)
        } else {
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        };

        // Distribute gain (put it all in first section)
        let scale = if i == 0 { remaining_gain } else { 1.0 };

        sections_data.extend_from_slice(&[b0 * scale, b1 * scale, b2 * scale, a0, a1, a2]);
    }

    let sections = Tensor::from_slice(&sections_data, &[n_sections, 6], device);
    Ok(SosFilter::new(sections))
}

/// Convert second-order sections to zeros, poles, gain (CPU-only).
///
/// This involves quadratic root finding for each section.
fn sos2zpk_cpu(sos: &SosFilter<CpuRuntime>) -> Result<ZpkFilter<CpuRuntime>> {
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

    // Get all section data at once (API boundary)
    let sections_data: Vec<f64> = sos.sections.to_vec();

    let mut all_zeros_re = Vec::new();
    let mut all_zeros_im = Vec::new();
    let mut all_poles_re = Vec::new();
    let mut all_poles_im = Vec::new();
    let mut gain = 1.0;

    for i in 0..n_sections {
        let offset = i * 6;
        let (b0, b1, b2) = (
            sections_data[offset],
            sections_data[offset + 1],
            sections_data[offset + 2],
        );
        let (a0, a1, a2) = (
            sections_data[offset + 3],
            sections_data[offset + 4],
            sections_data[offset + 5],
        );

        // Accumulate gain from numerator leading coefficient
        gain *= b0 / a0;

        // Find zeros of numerator (roots of b0 + b1*z^-1 + b2*z^-2)
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
            complex_poles.push((poles_re[i], poles_im[i].abs()));
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
fn biquad_coeffs(
    z1: (f64, f64),
    z2: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let (z1_re, z1_im) = z1;
    let (z2_re, z2_im) = z2;

    let b0 = 1.0;
    let b1 = -(z1_re + z2_re);
    let b2 = z1_re * z2_re - z1_im * z2_im;

    let (p1_re, p1_im) = p1;
    let (p2_re, p2_im) = p2;

    let a0 = 1.0;
    let a1 = -(p1_re + p2_re);
    let a2 = p1_re * p2_re - p1_im * p2_im;

    (b0, b1, b2, a0, a1, a2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_tf2zpk_simple() {
        let (client, device) = setup();

        // Simple lowpass: H(z) = 1 / (1 - 0.5z^-1)
        // Pole at z = 0.5
        use numr::tensor::Tensor;

        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device);
        let tf = TransferFunction::new(b, a);

        let zpk = client.tf2zpk(&tf).unwrap();

        assert_eq!(zpk.num_zeros(), 0);
        assert_eq!(zpk.num_poles(), 1);

        let poles_re: Vec<f64> = zpk.poles_real.to_vec();
        assert!((poles_re[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_zpk2tf_roundtrip() {
        let (client, device) = setup();
        use numr::tensor::Tensor;

        // Create a simple filter
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.0], &[2], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.9], &[2], &device);
        let tf_orig = TransferFunction::new(b, a);

        // Convert to ZPK and back
        let zpk = client.tf2zpk(&tf_orig).unwrap();
        let tf_back = client.zpk2tf(&zpk).unwrap();

        let b_orig: Vec<f64> = tf_orig.b.to_vec();
        let b_back: Vec<f64> = tf_back.b.to_vec();

        // Normalize for comparison
        let scale = b_orig[0] / b_back[0];
        for i in 0..b_orig.len() {
            assert!((b_orig[i] - b_back[i] * scale).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tf2sos() {
        let (client, device) = setup();
        use numr::tensor::Tensor;

        // 4th order filter (2 biquad sections)
        let b = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.4, 0.6, 0.4, 0.1], &[5], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0, -0.8, 0.64, -0.256, 0.0625], &[5], &device);
        let tf = TransferFunction::new(b, a);

        let sos = client.tf2sos(&tf, None).unwrap();

        // Should have 2 sections for 4th order
        assert_eq!(sos.num_sections(), 2);
        assert_eq!(sos.sections.shape(), &[2, 6]);
    }
}
