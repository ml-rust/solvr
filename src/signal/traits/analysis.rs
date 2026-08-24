//! Signal analysis algorithm traits.
//!
//! Provides algorithms for signal analysis and transformation.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Signal analysis algorithms.
///
/// All backends implementing signal analysis MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait SignalAnalysisAlgorithms<R: Runtime<DType = DType>> {
    /// Compute the analytic signal using the Hilbert transform.
    ///
    /// # Algorithm
    ///
    /// The analytic signal is computed via FFT:
    /// ```text
    /// X = FFT(x)
    /// H[0] = X[0]           (DC component)
    /// H[1:N/2] = 2 * X[1:N/2]  (positive frequencies, doubled)
    /// H[N/2:] = 0           (negative frequencies, zeroed)
    /// analytic = IFFT(H)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal (real)
    ///
    /// # Returns
    ///
    /// [`HilbertResult`] containing real and imaginary parts of analytic signal.
    fn hilbert(&self, x: &Tensor<R>) -> Result<HilbertResult<R>>;

    /// Resample a signal using polyphase filtering.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `num` - Numerator of resampling factor (upsample by num)
    /// * `den` - Denominator of resampling factor (downsample by den)
    ///
    /// # Returns
    ///
    /// Resampled signal with length approximately `len(x) * num / den`.
    fn resample(&self, x: &Tensor<R>, num: usize, den: usize) -> Result<Tensor<R>>;

    /// Downsample after applying an anti-aliasing filter.
    ///
    /// # Algorithm
    ///
    /// 1. Apply low-pass filter to prevent aliasing
    /// 2. Downsample by taking every q-th sample
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `q` - Downsampling factor
    /// * `params` - Decimation parameters
    ///
    /// # Returns
    ///
    /// Decimated signal with length `ceil(len(x) / q)`.
    fn decimate(&self, x: &Tensor<R>, q: usize, params: DecimateParams) -> Result<Tensor<R>>;

    /// Find peaks in a signal.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `params` - Peak finding parameters
    ///
    /// # Returns
    ///
    /// [`PeakResult`] containing peak indices and properties.
    fn find_peaks(&self, x: &Tensor<R>, params: PeakParams) -> Result<PeakResult<R>>;

    /// Apply Savitzky-Golay filter for smoothing.
    ///
    /// # Algorithm
    ///
    /// Fits successive sub-sets of data with a low-degree polynomial by
    /// the method of linear least squares, then uses the polynomial to
    /// estimate the central point.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `window_length` - Length of the filter window (must be odd and > polyorder)
    /// * `polyorder` - Order of the polynomial used to fit the samples
    /// * `deriv` - Derivative order (0 = smoothing)
    ///
    /// # Returns
    ///
    /// Filtered signal.
    fn savgol_filter(
        &self,
        x: &Tensor<R>,
        window_length: usize,
        polyorder: usize,
        deriv: usize,
    ) -> Result<Tensor<R>>;
}

/// Result from Hilbert transform.
#[derive(Debug, Clone)]
pub struct HilbertResult<R: Runtime<DType = DType>> {
    /// Real part (original signal).
    pub real: Tensor<R>,
    /// Imaginary part (Hilbert transform).
    pub imag: Tensor<R>,
}

impl<R: Runtime<DType = DType>> HilbertResult<R> {
    /// Get instantaneous amplitude (envelope).
    ///
    /// Computes `sqrt(real² + imag²)`.
    pub fn envelope(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.real.to_vec();
        let im: Vec<f64> = self.imag.to_vec();
        let n = re.len();

        let env: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect();

        let device = self.real.device();
        Tensor::from_slice(&env, &[n], device)
    }

    /// Get instantaneous phase in radians.
    ///
    /// Computes `atan2(imag, real)`.
    pub fn instantaneous_phase(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.real.to_vec();
        let im: Vec<f64> = self.imag.to_vec();
        let n = re.len();

        let phase: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| i.atan2(r))
            .collect();

        let device = self.real.device();
        Tensor::from_slice(&phase, &[n], device)
    }

    /// Get instantaneous frequency in radians per sample.
    ///
    /// Computes the derivative of the unwrapped phase.
    pub fn instantaneous_frequency(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.real.to_vec();
        let im: Vec<f64> = self.imag.to_vec();
        let n = re.len();

        // Compute phase
        let phase: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| i.atan2(r))
            .collect();

        // Compute unwrapped phase derivative
        let mut freq = Vec::with_capacity(n);
        freq.push(0.0); // First sample has no derivative

        for i in 1..n {
            let mut diff = phase[i] - phase[i - 1];
            // Unwrap
            while diff > std::f64::consts::PI {
                diff -= 2.0 * std::f64::consts::PI;
            }
            while diff < -std::f64::consts::PI {
                diff += 2.0 * std::f64::consts::PI;
            }
            freq.push(diff);
        }

        let device = self.real.device();
        Tensor::from_slice(&freq, &[n], device)
    }
}

/// Parameters for decimation.
#[derive(Debug, Clone)]
pub struct DecimateParams {
    /// Order of the lowpass filter (default: 8).
    pub n: usize,
    /// Filter implementation type (IIR or FIR).
    pub ftype: DecimateFilterImpl,
    /// Apply zero-phase filtering.
    pub zero_phase: bool,
}

impl Default for DecimateParams {
    fn default() -> Self {
        Self {
            n: 8,
            ftype: DecimateFilterImpl::Iir,
            zero_phase: true,
        }
    }
}

/// Filter implementation type for decimation.
///
/// Determines whether to use an IIR (Infinite Impulse Response) or
/// FIR (Finite Impulse Response) anti-aliasing filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecimateFilterImpl {
    /// Use IIR Chebyshev Type I filter (more efficient, slight phase distortion).
    #[default]
    Iir,
    /// Use FIR filter (linear phase, requires more computation).
    Fir,
}

/// Parameters for peak finding.
#[derive(Debug, Clone, Default)]
pub struct PeakParams {
    /// Minimum peak height.
    pub height: Option<f64>,
    /// Minimum height difference from neighbors.
    pub threshold: Option<f64>,
    /// Minimum horizontal distance between peaks.
    pub distance: Option<usize>,
    /// Minimum prominence.
    pub prominence: Option<f64>,
    /// Minimum peak width.
    pub width: Option<f64>,
}

impl PeakParams {
    /// Create default peak parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum height.
    pub fn with_height(mut self, height: f64) -> Self {
        self.height = Some(height);
        self
    }

    /// Set minimum distance between peaks.
    pub fn with_distance(mut self, distance: usize) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Set minimum prominence.
    pub fn with_prominence(mut self, prominence: f64) -> Self {
        self.prominence = Some(prominence);
        self
    }
}

/// Result from peak finding.
#[derive(Debug, Clone)]
pub struct PeakResult<R: Runtime<DType = DType>> {
    /// Indices of detected peaks.
    pub indices: Vec<usize>,
    /// Heights at peak locations.
    pub heights: Tensor<R>,
    /// Prominences of peaks (if computed).
    pub prominences: Option<Tensor<R>>,
}
