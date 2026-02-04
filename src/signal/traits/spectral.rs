//! Spectral analysis algorithm traits.
//!
//! Provides algorithms for power spectral density estimation.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Spectral analysis algorithms.
///
/// All backends implementing spectral analysis MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait SpectralAnalysisAlgorithms<R: Runtime> {
    /// Estimate power spectral density using Welch's method.
    ///
    /// # Algorithm
    ///
    /// 1. Divide signal into overlapping segments
    /// 2. Apply window to each segment
    /// 3. Compute periodogram of each segment
    /// 4. Average the periodograms
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `params` - Welch parameters (window, segment length, overlap, etc.)
    ///
    /// # Returns
    ///
    /// [`WelchResult`] containing frequencies and PSD estimate.
    fn welch(&self, x: &Tensor<R>, params: WelchParams<R>) -> Result<WelchResult<R>>;

    /// Estimate power spectral density using a simple periodogram.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Pxx = |FFT(x * window)|² / (fs * sum(window²))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `params` - Periodogram parameters
    ///
    /// # Returns
    ///
    /// [`PeriodogramResult`] containing frequencies and PSD.
    fn periodogram(
        &self,
        x: &Tensor<R>,
        params: PeriodogramParams<R>,
    ) -> Result<PeriodogramResult<R>>;

    /// Estimate cross spectral density using Welch's method.
    ///
    /// # Algorithm
    ///
    /// Similar to Welch, but computes cross-spectrum:
    /// ```text
    /// Pxy = conj(FFT(x)) * FFT(y)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - First input signal
    /// * `y` - Second input signal
    /// * `params` - Welch parameters
    ///
    /// # Returns
    ///
    /// [`CsdResult`] containing frequencies and complex cross-spectral density.
    fn csd(&self, x: &Tensor<R>, y: &Tensor<R>, params: WelchParams<R>) -> Result<CsdResult<R>>;

    /// Compute magnitude squared coherence between two signals.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Cxy = |Pxy|² / (Pxx * Pyy)
    /// ```
    ///
    /// Values range from 0 to 1, where 1 indicates perfect linear relationship.
    ///
    /// # Arguments
    ///
    /// * `x` - First input signal
    /// * `y` - Second input signal
    /// * `params` - Welch parameters
    ///
    /// # Returns
    ///
    /// [`CoherenceResult`] containing frequencies and coherence values.
    fn coherence(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        params: WelchParams<R>,
    ) -> Result<CoherenceResult<R>>;

    /// Compute Lomb-Scargle periodogram for unevenly sampled data.
    ///
    /// # Algorithm
    ///
    /// The Lomb-Scargle periodogram handles non-uniform sampling by fitting
    /// sinusoids at each frequency. It's equivalent to least-squares fitting.
    ///
    /// # Arguments
    ///
    /// * `t` - Sample times
    /// * `x` - Signal values at sample times
    /// * `freqs` - Frequencies at which to compute periodogram
    /// * `normalize` - If true, normalize by variance
    ///
    /// # Returns
    ///
    /// Power spectral density at the specified frequencies.
    fn lombscargle(
        &self,
        t: &Tensor<R>,
        x: &Tensor<R>,
        freqs: &Tensor<R>,
        normalize: bool,
    ) -> Result<Tensor<R>>;
}

/// Window type for spectral analysis.
#[derive(Debug, Clone)]
pub enum SpectralWindow<R: Runtime> {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Kaiser window with beta parameter.
    Kaiser(f64),
    /// Custom window coefficients.
    Custom(Tensor<R>),
}

#[allow(clippy::derivable_impls)]
impl<R: Runtime> Default for SpectralWindow<R> {
    fn default() -> Self {
        SpectralWindow::Hann
    }
}

/// Scaling mode for PSD estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PsdScaling {
    /// Power spectral density (V²/Hz).
    #[default]
    Density,
    /// Power spectrum (V²).
    Spectrum,
}

/// Detrend mode for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Detrend {
    /// No detrending.
    #[default]
    None,
    /// Remove mean (constant detrend).
    Constant,
    /// Remove linear trend.
    Linear,
}

/// Parameters for Welch's method.
#[derive(Debug, Clone)]
pub struct WelchParams<R: Runtime> {
    /// Sampling frequency in Hz (default: 1.0).
    pub fs: f64,
    /// Window type (default: Hann).
    pub window: SpectralWindow<R>,
    /// Length of each segment in samples.
    /// If None, defaults to 256.
    pub nperseg: Option<usize>,
    /// Number of overlapping samples between segments.
    /// If None, defaults to nperseg/2.
    pub noverlap: Option<usize>,
    /// FFT length. If None, defaults to nperseg.
    pub nfft: Option<usize>,
    /// Detrending mode.
    pub detrend: Detrend,
    /// Scaling mode.
    pub scaling: PsdScaling,
    /// If true, return one-sided spectrum for real signals (default: true).
    pub onesided: bool,
    /// Device for output tensors.
    pub device: R::Device,
}

impl<R: Runtime> WelchParams<R> {
    /// Create default Welch parameters with the given device.
    pub fn new(device: R::Device) -> Self {
        Self {
            fs: 1.0,
            window: SpectralWindow::default(),
            nperseg: None,
            noverlap: None,
            nfft: None,
            detrend: Detrend::default(),
            scaling: PsdScaling::default(),
            onesided: true,
            device,
        }
    }

    /// Set sampling frequency.
    pub fn with_fs(mut self, fs: f64) -> Self {
        self.fs = fs;
        self
    }

    /// Set window type.
    pub fn with_window(mut self, window: SpectralWindow<R>) -> Self {
        self.window = window;
        self
    }

    /// Set segment length.
    pub fn with_nperseg(mut self, nperseg: usize) -> Self {
        self.nperseg = Some(nperseg);
        self
    }

    /// Set overlap length.
    pub fn with_noverlap(mut self, noverlap: usize) -> Self {
        self.noverlap = Some(noverlap);
        self
    }

    /// Set FFT length.
    pub fn with_nfft(mut self, nfft: usize) -> Self {
        self.nfft = Some(nfft);
        self
    }
}

/// Parameters for periodogram.
#[derive(Debug, Clone)]
pub struct PeriodogramParams<R: Runtime> {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Window type.
    pub window: SpectralWindow<R>,
    /// FFT length. If None, uses signal length.
    pub nfft: Option<usize>,
    /// Detrending mode.
    pub detrend: Detrend,
    /// Scaling mode.
    pub scaling: PsdScaling,
    /// If true, return one-sided spectrum for real signals.
    pub onesided: bool,
    /// Device for output tensors.
    pub device: R::Device,
}

impl<R: Runtime> PeriodogramParams<R> {
    /// Create default periodogram parameters with the given device.
    pub fn new(device: R::Device) -> Self {
        Self {
            fs: 1.0,
            window: SpectralWindow::default(),
            nfft: None,
            detrend: Detrend::default(),
            scaling: PsdScaling::default(),
            onesided: true,
            device,
        }
    }

    /// Set sampling frequency.
    pub fn with_fs(mut self, fs: f64) -> Self {
        self.fs = fs;
        self
    }

    /// Set window type.
    pub fn with_window(mut self, window: SpectralWindow<R>) -> Self {
        self.window = window;
        self
    }
}

/// Result from Welch PSD estimation.
#[derive(Debug, Clone)]
pub struct WelchResult<R: Runtime> {
    /// Frequencies in Hz.
    pub freqs: Tensor<R>,
    /// Power spectral density.
    pub psd: Tensor<R>,
}

/// Result from periodogram.
#[derive(Debug, Clone)]
pub struct PeriodogramResult<R: Runtime> {
    /// Frequencies in Hz.
    pub freqs: Tensor<R>,
    /// Power spectral density.
    pub psd: Tensor<R>,
}

/// Result from cross spectral density.
#[derive(Debug, Clone)]
pub struct CsdResult<R: Runtime> {
    /// Frequencies in Hz.
    pub freqs: Tensor<R>,
    /// Cross spectral density (real part).
    pub pxy_real: Tensor<R>,
    /// Cross spectral density (imaginary part).
    pub pxy_imag: Tensor<R>,
}

impl<R: Runtime> CsdResult<R> {
    /// Get magnitude of cross spectral density.
    pub fn magnitude(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.pxy_real.to_vec();
        let im: Vec<f64> = self.pxy_imag.to_vec();
        let n = re.len();

        let mag: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect();

        let device = self.pxy_real.device();
        Ok(Tensor::from_slice(&mag, &[n], device))
    }

    /// Get phase of cross spectral density in radians.
    pub fn phase(&self) -> Result<Tensor<R>> {
        let re: Vec<f64> = self.pxy_real.to_vec();
        let im: Vec<f64> = self.pxy_imag.to_vec();
        let n = re.len();

        let phase: Vec<f64> = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| i.atan2(r))
            .collect();

        let device = self.pxy_real.device();
        Ok(Tensor::from_slice(&phase, &[n], device))
    }
}

/// Result from coherence estimation.
#[derive(Debug, Clone)]
pub struct CoherenceResult<R: Runtime> {
    /// Frequencies in Hz.
    pub freqs: Tensor<R>,
    /// Magnitude squared coherence (0 to 1).
    pub cxy: Tensor<R>,
}
