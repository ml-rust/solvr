//! IIR filter design traits.
//!
//! Provides design functions for Infinite Impulse Response (IIR) digital filters
//! using the bilinear transform method.

// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]
// Allow large enum variant size difference (ZpkFilter is larger than tf/sos)
#![allow(clippy::large_enum_variant)]

use crate::signal::filter::types::{
    AnalogPrototype, FilterOutput, FilterType, SosFilter, TransferFunction, ZpkFilter,
};
use numr::error::Result;
use numr::runtime::Runtime;

/// IIR filter design algorithms.
///
/// All backends implementing IIR design MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Design Method
///
/// All IIR filter design functions use the bilinear transform:
/// 1. Design analog prototype (lowpass, cutoff = 1 rad/s)
/// 2. Apply frequency transformation (LP→HP, LP→BP, LP→BS)
/// 3. Apply bilinear transform (s-plane → z-plane)
pub trait IirDesignAlgorithms<R: Runtime> {
    /// Design a Butterworth digital filter.
    ///
    /// The Butterworth filter has maximally flat passband response.
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order (number of poles)
    /// * `wn` - Critical frequency (normalized, 0 < Wn < 1 for digital)
    ///   - For lowpass/highpass: single frequency
    ///   - For bandpass/bandstop: [low, high] frequencies
    /// * `filter_type` - Type of filter (lowpass, highpass, bandpass, bandstop)
    /// * `output` - Output format (Ba, Zpk, or Sos)
    ///
    /// # Returns
    ///
    /// Filter in requested format wrapped in [`IirDesignResult`].
    fn butter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;

    /// Design a Chebyshev Type I digital filter.
    ///
    /// Chebyshev Type I has equiripple in the passband and monotonic rolloff.
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order
    /// * `rp` - Maximum ripple in passband (dB)
    /// * `wn` - Critical frequency (normalized)
    /// * `filter_type` - Type of filter
    /// * `output` - Output format
    fn cheby1(
        &self,
        order: usize,
        rp: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;

    /// Design a Chebyshev Type II digital filter.
    ///
    /// Chebyshev Type II has monotonic passband and equiripple in stopband.
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order
    /// * `rs` - Minimum attenuation in stopband (dB)
    /// * `wn` - Critical frequency (normalized)
    /// * `filter_type` - Type of filter
    /// * `output` - Output format
    fn cheby2(
        &self,
        order: usize,
        rs: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;

    /// Design an elliptic (Cauer) digital filter.
    ///
    /// Elliptic filters have equiripple in both passband and stopband,
    /// achieving the sharpest transition for a given order.
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order
    /// * `rp` - Maximum ripple in passband (dB)
    /// * `rs` - Minimum attenuation in stopband (dB)
    /// * `wn` - Critical frequency (normalized)
    /// * `filter_type` - Type of filter
    /// * `output` - Output format
    fn ellip(
        &self,
        order: usize,
        rp: f64,
        rs: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;

    /// Design a Bessel-Thomson digital filter.
    ///
    /// Bessel filters have maximally flat group delay (linear phase).
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order
    /// * `wn` - Critical frequency (normalized)
    /// * `filter_type` - Type of filter
    /// * `output` - Output format
    /// * `norm` - Normalization type (default: Phase)
    fn bessel(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        norm: Option<BesselNorm>,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;

    // ========================================================================
    // Prototype generation (internal, but exposed for advanced use)
    // ========================================================================

    /// Generate Butterworth analog prototype poles.
    ///
    /// Poles are evenly spaced on the left half of the unit circle.
    fn buttap(&self, order: usize, device: &R::Device) -> Result<AnalogPrototype<R>>;

    /// Generate Chebyshev Type I analog prototype.
    fn cheb1ap(&self, order: usize, rp: f64, device: &R::Device) -> Result<AnalogPrototype<R>>;

    /// Generate Chebyshev Type II analog prototype.
    fn cheb2ap(&self, order: usize, rs: f64, device: &R::Device) -> Result<AnalogPrototype<R>>;

    /// Generate elliptic analog prototype.
    fn ellipap(
        &self,
        order: usize,
        rp: f64,
        rs: f64,
        device: &R::Device,
    ) -> Result<AnalogPrototype<R>>;

    /// Generate Bessel analog prototype.
    fn besselap(
        &self,
        order: usize,
        norm: BesselNorm,
        device: &R::Device,
    ) -> Result<AnalogPrototype<R>>;

    // ========================================================================
    // Transformations
    // ========================================================================

    /// Apply bilinear transform to convert analog filter to digital.
    ///
    /// Maps s-plane to z-plane using:
    /// ```text
    /// s = (2/T) * (z - 1) / (z + 1)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `analog` - Analog prototype
    /// * `fs` - Sample rate (for frequency warping)
    fn bilinear_zpk(&self, analog: &AnalogPrototype<R>, fs: f64) -> Result<ZpkFilter<R>>;

    /// Transform lowpass prototype to target filter type.
    ///
    /// Applies s-domain frequency transformation:
    /// - LP→HP: s → ω₀/s
    /// - LP→BP: s → (s² + ω₀²)/(B·s)
    /// - LP→BS: s → B·s/(s² + ω₀²)
    fn lp2lp_zpk(&self, zpk: &AnalogPrototype<R>, wo: f64) -> Result<AnalogPrototype<R>>;
    fn lp2hp_zpk(&self, zpk: &AnalogPrototype<R>, wo: f64) -> Result<AnalogPrototype<R>>;
    fn lp2bp_zpk(&self, zpk: &AnalogPrototype<R>, wo: f64, bw: f64) -> Result<AnalogPrototype<R>>;
    fn lp2bs_zpk(&self, zpk: &AnalogPrototype<R>, wo: f64, bw: f64) -> Result<AnalogPrototype<R>>;
}

/// Result from IIR filter design functions.
///
/// Contains the filter in the requested output format.
#[derive(Debug, Clone)]
pub enum IirDesignResult<R: Runtime> {
    /// Transfer function coefficients (b, a).
    Ba(TransferFunction<R>),
    /// Zeros, poles, and gain.
    Zpk(ZpkFilter<R>),
    /// Second-order sections.
    Sos(SosFilter<R>),
}

impl<R: Runtime> IirDesignResult<R> {
    /// Get as transfer function, if that's the format.
    pub fn as_ba(&self) -> Option<&TransferFunction<R>> {
        match self {
            IirDesignResult::Ba(tf) => Some(tf),
            _ => None,
        }
    }

    /// Get as ZPK, if that's the format.
    pub fn as_zpk(&self) -> Option<&ZpkFilter<R>> {
        match self {
            IirDesignResult::Zpk(zpk) => Some(zpk),
            _ => None,
        }
    }

    /// Get as SOS, if that's the format.
    pub fn as_sos(&self) -> Option<&SosFilter<R>> {
        match self {
            IirDesignResult::Sos(sos) => Some(sos),
            _ => None,
        }
    }
}

/// Normalization type for Bessel filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BesselNorm {
    /// Normalize for maximally flat group delay at ω=0 (default).
    #[default]
    Phase,
    /// Normalize for -3dB at cutoff frequency.
    Delay,
    /// Normalize for magnitude = 1/√2 at cutoff.
    Mag,
}
