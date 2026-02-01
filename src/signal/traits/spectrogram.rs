//! Spectrogram trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithmic contract for spectrogram computation.
///
/// All backends implementing spectrogram algorithms MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait SpectrogramAlgorithms<R: Runtime> {
    /// Compute power spectrogram from signal.
    ///
    /// A spectrogram is the magnitude of the STFT raised to a power.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal tensor
    /// * `n_fft` - FFT size
    /// * `hop_length` - Hop between frames
    /// * `window` - Window function
    /// * `power` - Exponent for magnitude (2.0 for power, 1.0 for amplitude)
    fn spectrogram(
        &self,
        signal: &Tensor<R>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<R>>,
        power: f64,
    ) -> Result<Tensor<R>>;
}
