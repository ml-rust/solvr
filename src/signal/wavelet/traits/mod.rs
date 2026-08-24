//! Wavelet transform algorithm traits.
use crate::DType;

use super::types::Wavelet;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Discrete wavelet transform algorithms.
pub trait DwtAlgorithms<R: Runtime<DType = DType>> {
    /// Compute single-level discrete wavelet transform.
    ///
    /// # Algorithm
    ///
    /// Convolve signal with low-pass and high-pass filters, then downsample by 2:
    /// ```text
    /// cA = downsample(conv(x, dec_lo), 2)
    /// cD = downsample(conv(x, dec_hi), 2)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `wavelet` - Wavelet to use
    /// * `mode` - Extension mode for boundaries
    ///
    /// # Returns
    ///
    /// [`DwtResult`] containing approximation and detail coefficients.
    fn dwt(&self, x: &Tensor<R>, wavelet: &Wavelet, mode: ExtensionMode) -> Result<DwtResult<R>>;

    /// Compute inverse discrete wavelet transform.
    ///
    /// # Algorithm
    ///
    /// Upsample and convolve with reconstruction filters:
    /// ```text
    /// x = conv(upsample(cA), rec_lo) + conv(upsample(cD), rec_hi)
    /// ```
    fn idwt(
        &self,
        coeffs: &DwtResult<R>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<R>>;

    /// Multi-level wavelet decomposition.
    ///
    /// Recursively applies DWT to approximation coefficients.
    fn wavedec(
        &self,
        x: &Tensor<R>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
        level: usize,
    ) -> Result<WavedecResult<R>>;

    /// Multi-level wavelet reconstruction.
    ///
    /// Inverse of wavedec.
    fn waverec(
        &self,
        coeffs: &WavedecResult<R>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<R>>;

    /// 2D discrete wavelet transform.
    fn dwt2(&self, x: &Tensor<R>, wavelet: &Wavelet, mode: ExtensionMode)
    -> Result<Dwt2dResult<R>>;

    /// Inverse 2D discrete wavelet transform.
    fn idwt2(
        &self,
        coeffs: &Dwt2dResult<R>,
        wavelet: &Wavelet,
        mode: ExtensionMode,
    ) -> Result<Tensor<R>>;
}

/// Continuous wavelet transform algorithms.
pub trait CwtAlgorithms<R: Runtime<DType = DType>> {
    /// Compute continuous wavelet transform.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// CWT(a, b) = (1/sqrt(a)) * ∫ x(t) * ψ*((t-b)/a) dt
    /// ```
    ///
    /// Computed via convolution with scaled/translated wavelet.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `scales` - Scales to compute
    /// * `wavelet` - Wavelet to use (must be CWT wavelet)
    ///
    /// # Returns
    ///
    /// [`CwtResult`] containing complex coefficients for each scale.
    fn cwt(&self, x: &Tensor<R>, scales: &Tensor<R>, wavelet: &Wavelet) -> Result<CwtResult<R>>;
}

/// Extension mode for wavelet transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtensionMode {
    /// Zero-padding.
    #[default]
    Zero,
    /// Constant extension (replicate boundary values).
    Constant,
    /// Symmetric extension (mirror at boundaries).
    Symmetric,
    /// Periodic extension.
    Periodic,
    /// Smooth extension (using polynomial extrapolation).
    Smooth,
}

/// Result from single-level DWT.
#[derive(Debug, Clone)]
pub struct DwtResult<R: Runtime<DType = DType>> {
    /// Approximation coefficients (low-pass).
    pub approx: Tensor<R>,
    /// Detail coefficients (high-pass).
    pub detail: Tensor<R>,
}

/// Result from multi-level wavelet decomposition.
#[derive(Debug, Clone)]
pub struct WavedecResult<R: Runtime<DType = DType>> {
    /// Approximation coefficients at the coarsest level.
    pub approx: Tensor<R>,
    /// Detail coefficients at each level (finest to coarsest).
    pub details: Vec<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> WavedecResult<R> {
    /// Get the number of decomposition levels.
    pub fn num_levels(&self) -> usize {
        self.details.len()
    }

    /// Get detail coefficients at a specific level (1-indexed, 1 = finest).
    pub fn detail(&self, level: usize) -> Option<&Tensor<R>> {
        if level > 0 && level <= self.details.len() {
            Some(&self.details[level - 1])
        } else {
            None
        }
    }
}

/// Result from 2D DWT.
#[derive(Debug, Clone)]
pub struct Dwt2dResult<R: Runtime<DType = DType>> {
    /// LL (low-low) - approximation.
    pub ll: Tensor<R>,
    /// LH (low-high) - horizontal detail.
    pub lh: Tensor<R>,
    /// HL (high-low) - vertical detail.
    pub hl: Tensor<R>,
    /// HH (high-high) - diagonal detail.
    pub hh: Tensor<R>,
}

/// Result from continuous wavelet transform.
#[derive(Debug, Clone)]
pub struct CwtResult<R: Runtime<DType = DType>> {
    /// CWT coefficients (real part), shape [num_scales, signal_length].
    pub coeffs_real: Tensor<R>,
    /// CWT coefficients (imaginary part), shape [num_scales, signal_length].
    pub coeffs_imag: Tensor<R>,
    /// Scales used.
    pub scales: Tensor<R>,
}

impl<R: Runtime<DType = DType>> CwtResult<R> {
    /// Get magnitude of CWT coefficients.
    pub fn magnitude(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.coeffs_real.to_vec();
        let im: Vec<f64> = self.coeffs_imag.to_vec();

        let mag: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect();

        let shape = self.coeffs_real.shape().to_vec();
        let device = self.coeffs_real.device();
        Tensor::from_slice(&mag, &shape, device)
    }

    /// Get phase of CWT coefficients.
    pub fn phase(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.coeffs_real.to_vec();
        let im: Vec<f64> = self.coeffs_imag.to_vec();

        let phase: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| i.atan2(r))
            .collect();

        let shape = self.coeffs_real.shape().to_vec();
        let device = self.coeffs_real.device();
        Tensor::from_slice(&phase, &shape, device)
    }
}
