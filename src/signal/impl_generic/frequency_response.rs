//! Generic implementations of frequency response algorithms.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::types::SosFilter;
use crate::signal::traits::frequency_response::{FreqzResult, FreqzSpec};
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Compute frequency response of a digital filter.
pub fn freqz_impl<R, C>(
    _client: &C,
    b: &Tensor<R>,
    a: &Tensor<R>,
    worN: FreqzSpec<R>,
    whole: bool,
    device: &R::Device,
) -> Result<FreqzResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    // Get frequencies
    let w: Vec<f64> = match worN {
        FreqzSpec::NumPoints(n) => {
            let end = if whole { 2.0 * PI } else { PI };
            (0..n).map(|i| i as f64 * end / n as f64).collect()
        }
        FreqzSpec::Frequencies(ref tensor) => tensor.to_vec(),
    };

    let n_freqs = w.len();

    // Get filter coefficients
    let b_data: Vec<f64> = b.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    let _nb = b_data.len();
    let _na = a_data.len();

    // Normalize by a[0]
    let a0 = a_data[0];
    if a0.abs() < 1e-30 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "Leading denominator coefficient cannot be zero".to_string(),
        });
    }

    let b_norm: Vec<f64> = b_data.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a_data.iter().map(|&x| x / a0).collect();

    // Evaluate H(e^{jω}) = B(e^{jω}) / A(e^{jω})
    let mut h_real = Vec::with_capacity(n_freqs);
    let mut h_imag = Vec::with_capacity(n_freqs);

    for &omega in &w {
        // Compute B(e^{jω}) = Σ b[k] * e^{-jωk}
        let mut b_re = 0.0;
        let mut b_im = 0.0;
        for (k, &bk) in b_norm.iter().enumerate() {
            let angle = -omega * k as f64;
            b_re += bk * angle.cos();
            b_im += bk * angle.sin();
        }

        // Compute A(e^{jω}) = Σ a[k] * e^{-jωk}
        let mut a_re = 0.0;
        let mut a_im = 0.0;
        for (k, &ak) in a_norm.iter().enumerate() {
            let angle = -omega * k as f64;
            a_re += ak * angle.cos();
            a_im += ak * angle.sin();
        }

        // H = B / A (complex division)
        let denom = a_re * a_re + a_im * a_im;
        if denom < 1e-30 {
            // Pole at this frequency
            h_real.push(f64::INFINITY);
            h_imag.push(0.0);
        } else {
            let re = (b_re * a_re + b_im * a_im) / denom;
            let im = (b_im * a_re - b_re * a_im) / denom;
            h_real.push(re);
            h_imag.push(im);
        }
    }

    Ok(FreqzResult {
        w: Tensor::from_slice(&w, &[n_freqs], device),
        h_real: Tensor::from_slice(&h_real, &[n_freqs], device),
        h_imag: Tensor::from_slice(&h_imag, &[n_freqs], device),
    })
}

/// Compute frequency response of a filter in SOS form.
pub fn sosfreqz_impl<R, C>(
    _client: &C,
    sos: &SosFilter<R>,
    worN: FreqzSpec<R>,
    whole: bool,
    device: &R::Device,
) -> Result<FreqzResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let n_sections = sos.num_sections();

    if n_sections == 0 {
        // Unity gain
        let w: Vec<f64> = match worN {
            FreqzSpec::NumPoints(n) => {
                let end = if whole { 2.0 * PI } else { PI };
                (0..n).map(|i| i as f64 * end / n as f64).collect()
            }
            FreqzSpec::Frequencies(ref tensor) => tensor.to_vec(),
        };
        let n_freqs = w.len();
        return Ok(FreqzResult {
            w: Tensor::from_slice(&w, &[n_freqs], device),
            h_real: Tensor::from_slice(&vec![1.0; n_freqs], &[n_freqs], device),
            h_imag: Tensor::from_slice(&vec![0.0; n_freqs], &[n_freqs], device),
        });
    }

    // Get frequencies
    let w: Vec<f64> = match worN {
        FreqzSpec::NumPoints(n) => {
            let end = if whole { 2.0 * PI } else { PI };
            (0..n).map(|i| i as f64 * end / n as f64).collect()
        }
        FreqzSpec::Frequencies(ref tensor) => tensor.to_vec(),
    };

    let n_freqs = w.len();

    // Get SOS coefficients
    let sos_data: Vec<f64> = sos.sections.to_vec();

    // Initialize H = 1 (unity)
    let mut h_real = vec![1.0; n_freqs];
    let mut h_imag = vec![0.0; n_freqs];

    // Multiply by each section's frequency response
    for section_idx in 0..n_sections {
        let offset = section_idx * 6;
        let b0 = sos_data[offset];
        let b1 = sos_data[offset + 1];
        let b2 = sos_data[offset + 2];
        let a0 = sos_data[offset + 3];
        let a1 = sos_data[offset + 4];
        let a2 = sos_data[offset + 5];

        // Normalize
        let b0 = b0 / a0;
        let b1 = b1 / a0;
        let b2 = b2 / a0;
        let a1 = a1 / a0;
        let a2 = a2 / a0;

        for (i, &omega) in w.iter().enumerate() {
            // Compute B(e^{jω}) for this section
            let cos1 = (-omega).cos();
            let sin1 = (-omega).sin();
            let cos2 = (-2.0 * omega).cos();
            let sin2 = (-2.0 * omega).sin();

            let b_re = b0 + b1 * cos1 + b2 * cos2;
            let b_im = b1 * sin1 + b2 * sin2;

            // Compute A(e^{jω}) for this section (a0 = 1 after normalization)
            let a_re = 1.0 + a1 * cos1 + a2 * cos2;
            let a_im = a1 * sin1 + a2 * sin2;

            // H_section = B / A
            let denom = a_re * a_re + a_im * a_im;
            let (h_sec_re, h_sec_im) = if denom < 1e-30 {
                (f64::INFINITY, 0.0)
            } else {
                let re = (b_re * a_re + b_im * a_im) / denom;
                let im = (b_im * a_re - b_re * a_im) / denom;
                (re, im)
            };

            // Multiply total H by this section
            let new_re = h_real[i] * h_sec_re - h_imag[i] * h_sec_im;
            let new_im = h_real[i] * h_sec_im + h_imag[i] * h_sec_re;
            h_real[i] = new_re;
            h_imag[i] = new_im;
        }
    }

    Ok(FreqzResult {
        w: Tensor::from_slice(&w, &[n_freqs], device),
        h_real: Tensor::from_slice(&h_real, &[n_freqs], device),
        h_imag: Tensor::from_slice(&h_imag, &[n_freqs], device),
    })
}

/// Compute group delay of a digital filter.
pub fn group_delay_impl<R, C>(
    _client: &C,
    b: &Tensor<R>,
    a: &Tensor<R>,
    w: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let w_data: Vec<f64> = w.to_vec();
    let b_data: Vec<f64> = b.to_vec();
    let a_data: Vec<f64> = a.to_vec();
    let device = w.device();

    let _nb = b_data.len();
    let _na = a_data.len();

    // Group delay = -d(phase)/dω
    // For numerical stability, use the formula:
    // τ_g(ω) = Re{(B'/B) - (A'/A)} where ' denotes derivative w.r.t. z

    let mut tau = Vec::with_capacity(w_data.len());

    for &omega in &w_data {
        // Compute B(e^{jω}) and B'(e^{jω}) (derivative w.r.t. z^{-1})
        let mut b_re = 0.0;
        let mut b_im = 0.0;
        let mut bp_re = 0.0;
        let mut bp_im = 0.0;

        for (k, &bk) in b_data.iter().enumerate() {
            let angle = -omega * k as f64;
            b_re += bk * angle.cos();
            b_im += bk * angle.sin();
            // R = Σ k*b[k]*e^{-jωk} where e^{-jωk} = cos(ωk) - j*sin(ωk)
            // R_re = Σ k*bk*cos(ωk), R_im = -Σ k*bk*sin(ωk)
            // With angle = -ωk: cos(angle) = cos(ωk), sin(angle) = -sin(ωk)
            bp_re += (k as f64) * bk * angle.cos(); // k*bk*cos(ωk)
            bp_im += (k as f64) * bk * angle.sin(); // -k*bk*sin(ωk)
        }

        // Compute A(e^{jω}) and A'(e^{jω})
        let mut a_re = 0.0;
        let mut a_im = 0.0;
        let mut ap_re = 0.0;
        let mut ap_im = 0.0;

        for (k, &ak) in a_data.iter().enumerate() {
            let angle = -omega * k as f64;
            a_re += ak * angle.cos();
            a_im += ak * angle.sin();
            ap_re += (k as f64) * ak * angle.cos();
            ap_im += (k as f64) * ak * angle.sin();
        }

        // τ_g = Re{B'/B - A'/A}
        let b_mag_sq = b_re * b_re + b_im * b_im;
        let a_mag_sq = a_re * a_re + a_im * a_im;

        if b_mag_sq < 1e-30 || a_mag_sq < 1e-30 {
            tau.push(0.0);
        } else {
            // B'/B = (bp_re + j*bp_im) / (b_re + j*b_im)
            // Re{B'/B} = (bp_re*b_re + bp_im*b_im) / |B|^2
            let bp_over_b_re = (bp_re * b_re + bp_im * b_im) / b_mag_sq;
            let ap_over_a_re = (ap_re * a_re + ap_im * a_im) / a_mag_sq;

            tau.push(bp_over_b_re - ap_over_a_re);
        }
    }

    Ok(Tensor::from_slice(&tau, &[tau.len()], device))
}
