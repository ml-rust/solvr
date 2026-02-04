//! Frequency response computation traits.
//!
//! Provides algorithms for computing digital filter frequency responses.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::types::SosFilter;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Frequency response algorithms.
///
/// All backends implementing frequency response MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait FrequencyResponseAlgorithms<R: Runtime> {
    /// Compute the frequency response of a digital filter.
    ///
    /// # Algorithm
    ///
    /// Evaluates the transfer function at points on the unit circle:
    /// ```text
    /// H(e^{jω}) = B(e^{jω}) / A(e^{jω})
    ///
    /// where:
    /// B(e^{jω}) = Σ b[k] * e^{-jωk}
    /// A(e^{jω}) = Σ a[k] * e^{-jωk}
    /// ```
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients
    /// * `a` - Denominator coefficients
    /// * `worN` - Either:
    ///   - Number of frequency points (usize) - computes at worN points from 0 to π
    ///   - Specific frequencies to evaluate (tensor of ω values)
    /// * `whole` - If true, compute from 0 to 2π (default: false, 0 to π)
    ///
    /// # Returns
    ///
    /// [`FreqzResult`] containing:
    /// - `w`: Normalized frequencies (0 to π or 0 to 2π)
    /// - `h`: Complex frequency response values
    fn freqz(
        &self,
        b: &Tensor<R>,
        a: &Tensor<R>,
        worN: FreqzSpec<R>,
        whole: bool,
        device: &R::Device,
    ) -> Result<FreqzResult<R>>;

    /// Compute the frequency response of a filter in SOS form.
    ///
    /// # Algorithm
    ///
    /// For each section, compute the frequency response and multiply:
    /// ```text
    /// H(e^{jω}) = H_1(e^{jω}) * H_2(e^{jω}) * ... * H_n(e^{jω})
    /// ```
    ///
    /// # Arguments
    ///
    /// * `sos` - Second-order sections filter
    /// * `worN` - Frequency specification
    /// * `whole` - Compute full circle (0 to 2π)
    ///
    /// # Returns
    ///
    /// Frequency response result.
    fn sosfreqz(
        &self,
        sos: &SosFilter<R>,
        worN: FreqzSpec<R>,
        whole: bool,
        device: &R::Device,
    ) -> Result<FreqzResult<R>>;

    /// Compute group delay of a digital filter.
    ///
    /// Group delay is the negative derivative of the phase response:
    /// ```text
    /// τ_g(ω) = -d(phase(H(e^{jω})))/dω
    /// ```
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients
    /// * `a` - Denominator coefficients
    /// * `w` - Frequencies at which to compute group delay
    ///
    /// # Returns
    ///
    /// Group delay in samples at each frequency.
    fn group_delay(&self, b: &Tensor<R>, a: &Tensor<R>, w: &Tensor<R>) -> Result<Tensor<R>>;
}

/// Specification for frequency points in freqz.
#[derive(Debug, Clone)]
pub enum FreqzSpec<R: Runtime> {
    /// Number of equally spaced frequency points.
    NumPoints(usize),
    /// Specific frequency values (normalized, in radians).
    Frequencies(Tensor<R>),
}

impl<R: Runtime> Default for FreqzSpec<R> {
    fn default() -> Self {
        FreqzSpec::NumPoints(512)
    }
}

/// Result from frequency response computation.
#[derive(Debug, Clone)]
pub struct FreqzResult<R: Runtime> {
    /// Normalized angular frequencies (radians/sample).
    pub w: Tensor<R>,
    /// Complex frequency response (magnitude and phase).
    /// For real filters, stored as interleaved [re, im, re, im, ...].
    pub h_real: Tensor<R>,
    pub h_imag: Tensor<R>,
}

impl<R: Runtime> FreqzResult<R> {
    /// Get magnitude response |H(ω)|.
    pub fn magnitude(&self) -> Result<Tensor<R>> {
        let h_re: Vec<f64> = self.h_real.to_vec();
        let h_im: Vec<f64> = self.h_imag.to_vec();
        let n = h_re.len();

        let mag: Vec<f64> = h_re
            .iter()
            .zip(h_im.iter())
            .map(|(&re, &im)| (re * re + im * im).sqrt())
            .collect();

        let device = self.h_real.device();
        Ok(Tensor::from_slice(&mag, &[n], device))
    }

    /// Get phase response angle(H(ω)) in radians.
    pub fn phase(&self) -> Result<Tensor<R>> {
        let h_re: Vec<f64> = self.h_real.to_vec();
        let h_im: Vec<f64> = self.h_imag.to_vec();
        let n = h_re.len();

        let phase: Vec<f64> = h_re
            .iter()
            .zip(h_im.iter())
            .map(|(&re, &im)| im.atan2(re))
            .collect();

        let device = self.h_real.device();
        Ok(Tensor::from_slice(&phase, &[n], device))
    }

    /// Get magnitude response in decibels: 20*log10(|H(ω)|).
    pub fn magnitude_db(&self) -> Result<Tensor<R>> {
        let h_re: Vec<f64> = self.h_real.to_vec();
        let h_im: Vec<f64> = self.h_imag.to_vec();
        let n = h_re.len();

        let mag_db: Vec<f64> = h_re
            .iter()
            .zip(h_im.iter())
            .map(|(&re, &im)| {
                let mag = (re * re + im * im).sqrt();
                20.0 * mag.max(1e-300).log10()
            })
            .collect();

        let device = self.h_real.device();
        Ok(Tensor::from_slice(&mag_db, &[n], device))
    }
}
