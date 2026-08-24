//! Analog frequency response traits.
//!
//! Provides algorithms for computing analog filter frequency responses in the s-domain.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Analog frequency response algorithms.
///
/// All backends implementing analog frequency response MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait AnalogResponseAlgorithms<R: Runtime<DType = DType>> {
    /// Compute the frequency response of an analog filter.
    ///
    /// # Algorithm
    ///
    /// Evaluates the transfer function at points on the imaginary axis:
    /// ```text
    /// H(jω) = B(jω) / A(jω)
    ///
    /// where:
    /// B(jω) = b[0]*(jω)^M + b[1]*(jω)^(M-1) + ... + b[M]
    /// A(jω) = a[0]*(jω)^N + a[1]*(jω)^(N-1) + ... + a[N]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients in descending power order
    /// * `a` - Denominator coefficients in descending power order
    /// * `worN` - Frequencies to evaluate (angular frequencies in rad/s)
    ///
    /// # Returns
    ///
    /// [`FreqsResult`] containing:
    /// - `w`: Angular frequencies (rad/s)
    /// - `h_real`: Real part of complex frequency response
    /// - `h_imag`: Imaginary part of complex frequency response
    fn freqs(
        &self,
        b: &Tensor<R>,
        a: &Tensor<R>,
        worN: &Tensor<R>,
        device: &R::Device,
    ) -> Result<FreqsResult<R>>;
}

/// Result from analog frequency response computation.
#[derive(Debug, Clone)]
pub struct FreqsResult<R: Runtime<DType = DType>> {
    /// Angular frequencies (rad/s).
    pub w: Tensor<R>,
    /// Real part of complex frequency response.
    pub h_real: Tensor<R>,
    /// Imaginary part of complex frequency response.
    pub h_imag: Tensor<R>,
}

impl<R: Runtime<DType = DType>> FreqsResult<R> {
    /// Get magnitude response |H(jω)|.
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
        Tensor::from_slice(&mag, &[n], device)
    }

    /// Get phase response angle(H(jω)) in radians.
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
        Tensor::from_slice(&phase, &[n], device)
    }

    /// Get magnitude response in decibels: 20*log10(|H(jω)|).
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
        Tensor::from_slice(&mag_db, &[n], device)
    }
}
