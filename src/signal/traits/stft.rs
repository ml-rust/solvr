//! Short-Time Fourier Transform trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithmic contract for STFT and inverse STFT operations.
///
/// All backends implementing STFT algorithms MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait StftAlgorithms<R: Runtime> {
    /// Short-Time Fourier Transform.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal tensor of shape [..., time]
    /// * `n_fft` - FFT size (must be power of 2)
    /// * `hop_length` - Number of samples between frames (default: n_fft / 4)
    /// * `window` - Window function tensor of shape [n_fft] (default: Hann window)
    /// * `center` - If true, pad signal so frame is centered at sample
    /// * `normalized` - If true, normalize by 1/sqrt(n_fft)
    ///
    /// # Returns
    ///
    /// Complex tensor of shape [..., n_frames, n_fft/2 + 1]
    fn stft(
        &self,
        signal: &Tensor<R>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<R>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<R>>;

    /// Inverse Short-Time Fourier Transform.
    ///
    /// Reconstructs the time-domain signal from STFT output using overlap-add.
    fn istft(
        &self,
        stft_matrix: &Tensor<R>,
        hop_length: Option<usize>,
        window: Option<&Tensor<R>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_num_frames() {
        let signal_len = 256;
        let n_fft = 256;
        let hop_length = 64;
        let center = true;

        let padded_len = if center {
            signal_len + n_fft
        } else {
            signal_len
        };

        let n_frames = if padded_len < n_fft {
            0
        } else {
            (padded_len - n_fft) / hop_length + 1
        };

        let expected = (256 + 256 - 256) / 64 + 1;
        assert_eq!(n_frames, expected);
    }
}
