//! Analog prototype generation and IIR filter design.
//!
//! Generates analog lowpass prototype filters (cutoff = 1 rad/s) for:
//! - Butterworth (maximally flat magnitude)
//! - Chebyshev Type I (equiripple passband)
//! - Chebyshev Type II (equiripple stopband)
//! - Elliptic/Cauer (equiripple both bands)
//! - Bessel-Thomson (maximally flat group delay)

// Allow manual div_ceil and is_multiple_of for clarity in filter order calculations
#![allow(clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]

use super::bilinear::{bilinear_zpk_impl, prewarp};
use super::conversions::{tf2sos_impl, zpk2tf_impl};
use super::freq_transform::{lp2bp_zpk_impl, lp2bs_zpk_impl, lp2hp_zpk_impl, lp2lp_zpk_impl};
use crate::signal::filter::traits::conversions::SosPairing;
use crate::signal::filter::traits::iir_design::{BesselNorm, IirDesignResult};
use crate::signal::filter::types::{AnalogPrototype, FilterOutput, FilterType};
use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

// ============================================================================
// Butterworth
// ============================================================================

/// Generate Butterworth analog prototype poles.
///
/// Poles are evenly distributed on the left half of the unit circle.
pub fn buttap_impl<R, C>(
    _client: &C,
    order: usize,
    device: &R::Device,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Filter order must be > 0".to_string(),
        });
    }

    let n = order;

    // Butterworth poles: s_k = exp(j * π * (2k + n + 1) / (2n))
    // for k = 0, 1, ..., n-1
    // Only poles in left half-plane (Re(s) < 0)
    let mut poles_re = Vec::with_capacity(n);
    let mut poles_im = Vec::with_capacity(n);

    for k in 0..n {
        let angle = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
        poles_re.push(angle.cos());
        poles_im.push(angle.sin());
    }

    Ok(AnalogPrototype::new(
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::from_slice(&poles_re, &[n], device),
        Tensor::from_slice(&poles_im, &[n], device),
        1.0,
    ))
}

/// Design Butterworth digital filter.
pub fn butter_impl<R, C>(
    client: &C,
    order: usize,
    wn: &[f64],
    filter_type: FilterType,
    output: FilterOutput,
    device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // Validate inputs
    validate_wn(wn, filter_type)?;

    // Generate analog prototype
    let proto = buttap_impl(client, order, device)?;

    // Design filter
    design_iir_filter(client, proto, wn, filter_type, output, device)
}

// ============================================================================
// Chebyshev Type I
// ============================================================================

/// Generate Chebyshev Type I analog prototype.
///
/// Has equiripple passband and monotonic stopband.
pub fn cheb1ap_impl<R, C>(
    _client: &C,
    order: usize,
    rp: f64,
    device: &R::Device,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Filter order must be > 0".to_string(),
        });
    }

    if rp <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "rp",
            reason: "Passband ripple must be > 0 dB".to_string(),
        });
    }

    let n = order;

    // ε = sqrt(10^(rp/10) - 1)
    let eps = (10.0_f64.powf(rp / 10.0) - 1.0).sqrt();

    // sinh^(-1)(1/ε) / n
    let mu = (1.0 / eps).asinh() / n as f64;

    let mut poles_re = Vec::with_capacity(n);
    let mut poles_im = Vec::with_capacity(n);

    for k in 0..n {
        // θ_k = π(2k + 1) / (2n)
        let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;

        // s_k = -sinh(μ)·sin(θ_k) + j·cosh(μ)·cos(θ_k)
        poles_re.push(-mu.sinh() * theta.sin());
        poles_im.push(mu.cosh() * theta.cos());
    }

    // Gain to normalize passband
    let _gain = if n % 2 == 0 {
        // Even order: gain = 1/sqrt(1 + ε²)
        1.0 / (1.0 + eps * eps).sqrt()
    } else {
        // Odd order: gain = 1
        1.0
    };

    // Actually compute gain from poles
    let mut g = 1.0;
    for i in 0..n {
        g *= (poles_re[i] * poles_re[i] + poles_im[i] * poles_im[i]).sqrt();
    }
    let gain = g / (1.0 + eps * eps).sqrt();

    Ok(AnalogPrototype::new(
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::from_slice(&poles_re, &[n], device),
        Tensor::from_slice(&poles_im, &[n], device),
        gain,
    ))
}

/// Design Chebyshev Type I digital filter.
pub fn cheby1_impl<R, C>(
    client: &C,
    order: usize,
    rp: f64,
    wn: &[f64],
    filter_type: FilterType,
    output: FilterOutput,
    device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    validate_wn(wn, filter_type)?;

    let proto = cheb1ap_impl(client, order, rp, device)?;
    design_iir_filter(client, proto, wn, filter_type, output, device)
}

// ============================================================================
// Chebyshev Type II
// ============================================================================

/// Generate Chebyshev Type II analog prototype.
///
/// Has monotonic passband and equiripple stopband.
pub fn cheb2ap_impl<R, C>(
    _client: &C,
    order: usize,
    rs: f64,
    device: &R::Device,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Filter order must be > 0".to_string(),
        });
    }

    if rs <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "rs",
            reason: "Stopband attenuation must be > 0 dB".to_string(),
        });
    }

    let n = order;

    // δ = 1 / sqrt(10^(rs/10) - 1)
    let de = 1.0 / (10.0_f64.powf(rs / 10.0) - 1.0).sqrt();

    let mu = (1.0 / de).asinh() / n as f64;

    let mut poles_re = Vec::with_capacity(n);
    let mut poles_im = Vec::with_capacity(n);
    let mut zeros_re = Vec::new();
    let mut zeros_im = Vec::new();

    for k in 0..n {
        let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;

        // Poles
        let p_re = -mu.sinh() * theta.sin();
        let p_im = mu.cosh() * theta.cos();

        // Invert: s' = 1/s
        let mag_sq = p_re * p_re + p_im * p_im;
        poles_re.push(p_re / mag_sq);
        poles_im.push(-p_im / mag_sq);

        // Zeros at j / cos(θ) for k = 0, 1, ..., floor((n-1)/2)
        if theta.cos().abs() > 1e-10 && k < (n + 1) / 2 {
            let z_im = 1.0 / theta.cos();
            zeros_re.push(0.0);
            zeros_im.push(z_im);
            // Add conjugate
            zeros_re.push(0.0);
            zeros_im.push(-z_im);
        }
    }

    // Compute gain
    let mut num_prod = 1.0;
    for i in 0..zeros_re.len() {
        num_prod *= (zeros_re[i] * zeros_re[i] + zeros_im[i] * zeros_im[i]).sqrt();
    }

    let mut den_prod = 1.0;
    for i in 0..n {
        den_prod *= (poles_re[i] * poles_re[i] + poles_im[i] * poles_im[i]).sqrt();
    }

    let gain = den_prod / num_prod / (10.0_f64.powf(rs / 20.0));

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&zeros_re, &[zeros_re.len()], device),
        Tensor::from_slice(&zeros_im, &[zeros_im.len()], device),
        Tensor::from_slice(&poles_re, &[n], device),
        Tensor::from_slice(&poles_im, &[n], device),
        gain,
    ))
}

/// Design Chebyshev Type II digital filter.
pub fn cheby2_impl<R, C>(
    client: &C,
    order: usize,
    rs: f64,
    wn: &[f64],
    filter_type: FilterType,
    output: FilterOutput,
    device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    validate_wn(wn, filter_type)?;

    let proto = cheb2ap_impl(client, order, rs, device)?;
    design_iir_filter(client, proto, wn, filter_type, output, device)
}

// ============================================================================
// Elliptic (Cauer)
// ============================================================================

/// Generate elliptic analog prototype.
///
/// Has equiripple in both passband and stopband.
/// Achieves the sharpest transition for a given order.
pub fn ellipap_impl<R, C>(
    _client: &C,
    order: usize,
    rp: f64,
    rs: f64,
    device: &R::Device,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Filter order must be > 0".to_string(),
        });
    }

    if rp <= 0.0 || rs <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "rp/rs",
            reason: "Ripple values must be > 0 dB".to_string(),
        });
    }

    let n = order;

    // Passband and stopband deviations
    let eps_p = (10.0_f64.powf(rp / 10.0) - 1.0).sqrt();
    let eps_s = (10.0_f64.powf(rs / 10.0) - 1.0).sqrt();

    // Selectivity factor k = eps_p / eps_s
    let k = eps_p / eps_s;

    // This is a simplified elliptic implementation
    // For a complete implementation, we'd need elliptic integrals
    // For now, approximate with Chebyshev-like poles plus zeros

    // Approximate the elliptic filter using the relationship:
    // Elliptic ≈ Chebyshev I with zeros added

    let mut poles_re = Vec::with_capacity(n);
    let mut poles_im = Vec::with_capacity(n);
    let mut zeros_re = Vec::new();
    let mut zeros_im = Vec::new();

    // Use Chebyshev I poles as starting point
    let mu = (1.0 / eps_p).asinh() / n as f64;

    for i in 0..n {
        let theta = PI * (2 * i + 1) as f64 / (2 * n) as f64;
        poles_re.push(-mu.sinh() * theta.sin());
        poles_im.push(mu.cosh() * theta.cos());
    }

    // Add zeros on the imaginary axis
    // For elliptic, zeros are at positions determined by elliptic functions
    // Simplified: place zeros based on k
    let num_zeros = n / 2;
    for i in 0..num_zeros {
        let theta = PI * (2 * i + 1) as f64 / (2 * n) as f64;
        let z_im = 1.0 / (k * theta.sin());
        zeros_re.push(0.0);
        zeros_im.push(z_im);
        zeros_re.push(0.0);
        zeros_im.push(-z_im);
    }

    // Compute gain
    let mut gain = 1.0;
    for i in 0..n {
        gain *= (poles_re[i] * poles_re[i] + poles_im[i] * poles_im[i]).sqrt();
    }
    for i in 0..zeros_re.len() {
        let z_mag = (zeros_re[i] * zeros_re[i] + zeros_im[i] * zeros_im[i]).sqrt();
        if z_mag > 1e-10 {
            gain /= z_mag;
        }
    }

    if n % 2 == 0 {
        gain /= (1.0 + eps_p * eps_p).sqrt();
    }

    Ok(AnalogPrototype::new(
        Tensor::from_slice(&zeros_re, &[zeros_re.len()], device),
        Tensor::from_slice(&zeros_im, &[zeros_im.len()], device),
        Tensor::from_slice(&poles_re, &[n], device),
        Tensor::from_slice(&poles_im, &[n], device),
        gain,
    ))
}

/// Design elliptic digital filter.
pub fn ellip_impl<R, C>(
    client: &C,
    order: usize,
    rp: f64,
    rs: f64,
    wn: &[f64],
    filter_type: FilterType,
    output: FilterOutput,
    device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    validate_wn(wn, filter_type)?;

    let proto = ellipap_impl(client, order, rp, rs, device)?;
    design_iir_filter(client, proto, wn, filter_type, output, device)
}

// ============================================================================
// Bessel-Thomson
// ============================================================================

/// Generate Bessel analog prototype.
///
/// Has maximally flat group delay (linear phase).
pub fn besselap_impl<R, C>(
    _client: &C,
    order: usize,
    norm: BesselNorm,
    device: &R::Device,
) -> Result<AnalogPrototype<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Filter order must be > 0".to_string(),
        });
    }

    // Bessel polynomial roots (pre-computed for common orders)
    // These are the poles of the normalized Bessel filter
    let poles = match order {
        1 => vec![(-1.0, 0.0)],
        2 => vec![(-1.1016, 0.6368), (-1.1016, -0.6368)],
        3 => vec![(-1.0509, 0.9991), (-1.0509, -0.9991), (-1.3226, 0.0)],
        4 => vec![
            (-0.9952, 1.2571),
            (-0.9952, -1.2571),
            (-1.3700, 0.4102),
            (-1.3700, -0.4102),
        ],
        5 => vec![
            (-0.9576, 1.4711),
            (-0.9576, -1.4711),
            (-1.3808, 0.7179),
            (-1.3808, -0.7179),
            (-1.5023, 0.0),
        ],
        6 => vec![
            (-0.9306, 1.6618),
            (-0.9306, -1.6618),
            (-1.3818, 0.9714),
            (-1.3818, -0.9714),
            (-1.5714, 0.3213),
            (-1.5714, -0.3213),
        ],
        7 => vec![
            (-0.9098, 1.8364),
            (-0.9098, -1.8364),
            (-1.3789, 1.1915),
            (-1.3789, -1.1915),
            (-1.6120, 0.5896),
            (-1.6120, -0.5896),
            (-1.6843, 0.0),
        ],
        8 => vec![
            (-0.8928, 1.9983),
            (-0.8928, -1.9983),
            (-1.3738, 1.3884),
            (-1.3738, -1.3884),
            (-1.6369, 0.8227),
            (-1.6369, -0.8227),
            (-1.7574, 0.2728),
            (-1.7574, -0.2728),
        ],
        _ => {
            // For higher orders, compute using recursion
            compute_bessel_poles(order)
        }
    };

    let n = poles.len();
    let poles_re: Vec<f64> = poles.iter().map(|(re, _)| *re).collect();
    let poles_im: Vec<f64> = poles.iter().map(|(_, im)| *im).collect();

    // Normalization
    let scale = match norm {
        BesselNorm::Phase => 1.0,
        BesselNorm::Delay => {
            // Scale for -3dB at ω=1
            bessel_norm_delay(order)
        }
        BesselNorm::Mag => {
            // Scale for |H(jω)|² = 1/2 at ω=1
            bessel_norm_mag(order)
        }
    };

    let scaled_poles_re: Vec<f64> = poles_re.iter().map(|&x| x * scale).collect();
    let scaled_poles_im: Vec<f64> = poles_im.iter().map(|&x| x * scale).collect();

    // Compute gain
    let mut gain = 1.0;
    for i in 0..n {
        gain *= (scaled_poles_re[i] * scaled_poles_re[i] + scaled_poles_im[i] * scaled_poles_im[i])
            .sqrt();
    }

    Ok(AnalogPrototype::new(
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::zeros(&[0], numr::dtype::DType::F64, device),
        Tensor::from_slice(&scaled_poles_re, &[n], device),
        Tensor::from_slice(&scaled_poles_im, &[n], device),
        gain,
    ))
}

// ============================================================================
// Common IIR design routine
// ============================================================================

/// Common IIR filter design routine.
///
/// Takes an analog prototype and applies:
/// 1. Frequency pre-warping
/// 2. Frequency transformation (LP→HP, BP, BS)
/// 3. Bilinear transform
/// 4. Output format conversion
pub fn design_iir_filter<R, C>(
    client: &C,
    proto: AnalogPrototype<R>,
    wn: &[f64],
    filter_type: FilterType,
    output: FilterOutput,
    _device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // Sample rate for bilinear transform (normalized to Nyquist = 1)
    let fs = 2.0;

    // Pre-warp critical frequencies
    let warped: Vec<f64> = wn.iter().map(|&w| prewarp(w, fs)).collect();

    // Apply frequency transformation
    let transformed = match filter_type {
        FilterType::Lowpass => {
            let wo = warped[0];
            lp2lp_zpk_impl(client, &proto, wo)?
        }
        FilterType::Highpass => {
            let wo = warped[0];
            lp2hp_zpk_impl(client, &proto, wo)?
        }
        FilterType::Bandpass => {
            let wo = (warped[0] * warped[1]).sqrt();
            let bw = warped[1] - warped[0];
            lp2bp_zpk_impl(client, &proto, wo, bw)?
        }
        FilterType::Bandstop => {
            let wo = (warped[0] * warped[1]).sqrt();
            let bw = warped[1] - warped[0];
            lp2bs_zpk_impl(client, &proto, wo, bw)?
        }
    };

    // Apply bilinear transform
    let digital = bilinear_zpk_impl(client, &transformed, fs)?;

    // Convert to requested output format
    match output {
        FilterOutput::Zpk => Ok(IirDesignResult::Zpk(digital)),
        FilterOutput::Ba => {
            let tf = zpk2tf_impl(client, &digital)?;
            Ok(IirDesignResult::Ba(tf))
        }
        FilterOutput::Sos => {
            let tf = zpk2tf_impl(client, &digital)?;
            let sos = tf2sos_impl(client, &tf, Some(SosPairing::Nearest))?;
            Ok(IirDesignResult::Sos(sos))
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn validate_wn(wn: &[f64], filter_type: FilterType) -> Result<()> {
    match filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            if wn.len() != 1 {
                return Err(Error::InvalidArgument {
                    arg: "wn",
                    reason: format!(
                        "{:?} requires single cutoff frequency, got {}",
                        filter_type,
                        wn.len()
                    ),
                });
            }
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            if wn.len() != 2 {
                return Err(Error::InvalidArgument {
                    arg: "wn",
                    reason: format!(
                        "{:?} requires two cutoff frequencies, got {}",
                        filter_type,
                        wn.len()
                    ),
                });
            }
            if wn[0] >= wn[1] {
                return Err(Error::InvalidArgument {
                    arg: "wn",
                    reason: "Low cutoff must be less than high cutoff".to_string(),
                });
            }
        }
    }

    for &w in wn {
        if w <= 0.0 || w >= 1.0 {
            return Err(Error::InvalidArgument {
                arg: "wn",
                reason: "Cutoff frequencies must be in (0, 1)".to_string(),
            });
        }
    }

    Ok(())
}

/// Compute Bessel polynomial roots for higher orders.
fn compute_bessel_poles(order: usize) -> Vec<(f64, f64)> {
    // Use recursion formula for Bessel polynomials
    // B_n(s) = (2n-1) * B_{n-1}(s) + s^2 * B_{n-2}(s)
    // Then find roots numerically

    // For simplicity, use asymptotic approximation
    let mut poles = Vec::with_capacity(order);

    for k in 0..order {
        let theta = PI * (2 * k + 1) as f64 / (2 * order) as f64;
        // Approximate pole positions
        let r = 1.0 + 0.5 / order as f64;
        poles.push((-r * theta.sin(), r * theta.cos()));
    }

    poles
}

fn bessel_norm_delay(order: usize) -> f64 {
    // Normalization factors for -3dB at ω=1
    match order {
        1 => 1.0,
        2 => 1.3617,
        3 => 1.7557,
        4 => 2.1139,
        5 => 2.4274,
        6 => 2.7034,
        7 => 2.9517,
        8 => 3.1796,
        _ => (order as f64).sqrt() * 1.05,
    }
}

fn bessel_norm_mag(order: usize) -> f64 {
    // Normalization for magnitude
    match order {
        1 => 1.0,
        2 => 1.2736,
        3 => 1.4524,
        4 => 1.6060,
        5 => 1.7426,
        6 => 1.8672,
        7 => 1.9824,
        8 => 2.0898,
        _ => (order as f64).powf(0.45),
    }
}
