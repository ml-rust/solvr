//! FIR filter design traits.
//!
//! Provides design functions for Finite Impulse Response (FIR) digital filters
//! using windowed sinc and frequency sampling methods.

use crate::signal::filter::types::FilterType;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// FIR filter design algorithms.
///
/// All backends implementing FIR design MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait FirDesignAlgorithms<R: Runtime> {
    /// Design an FIR filter using the window method.
    ///
    /// # Algorithm
    ///
    /// 1. Design ideal (brick-wall) lowpass filter via sinc function
    /// 2. Apply frequency transformation for filter type
    /// 3. Apply window function to truncated impulse response
    ///
    /// # Arguments
    ///
    /// * `numtaps` - Number of filter coefficients (filter length)
    /// * `cutoff` - Cutoff frequency (normalized, 0 < f < 1)
    ///   - For lowpass/highpass: single frequency
    ///   - For bandpass/bandstop: [low, high] frequencies
    /// * `filter_type` - Type of filter
    /// * `window` - Window function name or custom window
    /// * `scale` - If true, scale so gain at center of passband is 1 (default: true)
    ///
    /// # Returns
    ///
    /// FIR filter coefficients [numtaps].
    fn firwin(
        &self,
        numtaps: usize,
        cutoff: &[f64],
        filter_type: FilterType,
        window: FirWindow,
        scale: bool,
        device: &R::Device,
    ) -> Result<Tensor<R>>;

    /// Design an FIR filter using frequency sampling method.
    ///
    /// # Algorithm
    ///
    /// 1. Specify desired frequency response at sample points
    /// 2. Apply inverse FFT to get impulse response
    /// 3. Apply window function
    ///
    /// # Arguments
    ///
    /// * `numtaps` - Number of filter coefficients
    /// * `freq` - Frequency points (normalized, 0 to 1)
    /// * `gain` - Desired gain at each frequency point
    /// * `antisymmetric` - If true, design Type III/IV filter (default: false)
    ///
    /// # Returns
    ///
    /// FIR filter coefficients [numtaps].
    fn firwin2(
        &self,
        numtaps: usize,
        freq: &[f64],
        gain: &[f64],
        antisymmetric: bool,
        window: FirWindow,
        device: &R::Device,
    ) -> Result<Tensor<R>>;

    /// Design a minimum-phase FIR filter from linear-phase prototype.
    ///
    /// # Algorithm
    ///
    /// 1. Start with linear-phase FIR (symmetric)
    /// 2. Compute cepstrum via log-FFT
    /// 3. Apply minimum-phase reconstruction
    ///
    /// # Arguments
    ///
    /// * `h` - Linear-phase FIR filter coefficients
    ///
    /// # Returns
    ///
    /// Minimum-phase FIR filter (half the length).
    fn minimum_phase(&self, h: &Tensor<R>) -> Result<Tensor<R>>;
}

/// Window specification for FIR filter design.
#[derive(Debug, Clone)]
pub enum FirWindow {
    /// Rectangular window (no windowing).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Kaiser window with specified beta parameter.
    Kaiser(f64),
    /// Custom window coefficients.
    Custom(Vec<f64>),
}

#[allow(clippy::derivable_impls)]
impl Default for FirWindow {
    fn default() -> Self {
        FirWindow::Hamming
    }
}

impl FirWindow {
    /// Estimate Kaiser window beta for given attenuation.
    ///
    /// # Arguments
    ///
    /// * `atten` - Desired stopband attenuation in dB
    ///
    /// # Returns
    ///
    /// Beta parameter for Kaiser window.
    pub fn kaiser_beta(atten: f64) -> f64 {
        if atten > 50.0 {
            0.1102 * (atten - 8.7)
        } else if atten > 21.0 {
            0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)
        } else {
            0.0
        }
    }

    /// Estimate filter order for Kaiser window.
    ///
    /// # Arguments
    ///
    /// * `ripple` - Passband ripple in dB
    /// * `atten` - Stopband attenuation in dB
    /// * `transition_width` - Normalized transition bandwidth
    ///
    /// # Returns
    ///
    /// Estimated filter order.
    pub fn kaiser_order(ripple: f64, atten: f64, transition_width: f64) -> usize {
        let a = atten.max(ripple);
        let n = (a - 7.95) / (2.285 * std::f64::consts::PI * transition_width);
        (n.ceil() as usize).max(1)
    }
}
