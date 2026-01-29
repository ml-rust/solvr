//! Signal processing algorithms
//!
//! This module provides signal processing operations including:
//! - 1D and 2D convolution (FFT-based)
//! - 1D and 2D cross-correlation
//! - STFT (Short-Time Fourier Transform)
//! - ISTFT (Inverse STFT)
//! - Spectrogram
//!
//! # Backend Support
//!
//! All operations are implemented for:
//! - CPU (F32, F64)
//! - CUDA (F32, F64) - requires `cuda` feature
//! - WebGPU (F32 only) - requires `wgpu` feature
//!
//! # Algorithm: FFT-based Convolution
//!
//! ```text
//! convolve(signal, kernel, mode):
//!
//! 1. Compute output length based on mode:
//!    - full: len(signal) + len(kernel) - 1
//!    - same: max(len(signal), len(kernel))
//!    - valid: |len(signal) - len(kernel)| + 1
//!
//! 2. Pad both to next power-of-2 >= (len(signal) + len(kernel) - 1)
//!
//! 3. FFT convolution:
//!    X = rfft(pad(signal, padded_len))
//!    H = rfft(pad(kernel, padded_len))
//!    Y = X * H  (element-wise complex multiply)
//!    result = irfft(Y, n=padded_len)
//!
//! 4. Slice output based on mode
//! ```

mod cpu;
mod stft_core;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::algorithm::fft::FftAlgorithms;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// ============================================================================
// Convolution Mode
// ============================================================================

/// Convolution output mode
///
/// Determines the size and alignment of the convolution output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvMode {
    /// Full convolution output
    ///
    /// Output length = N + M - 1, where N = signal length, M = kernel length.
    /// Contains all points where the kernel and signal overlap.
    #[default]
    Full,

    /// Same-size output
    ///
    /// Output length = max(N, M).
    /// Output is centered relative to the 'full' output.
    /// Matches scipy.signal.convolve behavior.
    Same,

    /// Valid convolution output
    ///
    /// Output length = max(N, M) - min(N, M) + 1.
    /// Contains only points where the kernel fits entirely within the signal.
    Valid,
}

impl ConvMode {
    /// Calculate output length for 1D convolution
    pub fn output_len(&self, signal_len: usize, kernel_len: usize) -> usize {
        match self {
            ConvMode::Full => signal_len + kernel_len - 1,
            ConvMode::Same => signal_len.max(kernel_len),
            ConvMode::Valid => {
                let min_len = signal_len.min(kernel_len);
                let max_len = signal_len.max(kernel_len);
                if min_len == 0 {
                    0
                } else {
                    max_len - min_len + 1
                }
            }
        }
    }

    /// Calculate start offset for slicing full convolution result
    pub fn slice_start(&self, signal_len: usize, kernel_len: usize) -> usize {
        match self {
            ConvMode::Full => 0,
            ConvMode::Same => {
                let full_len = signal_len + kernel_len - 1;
                let out_len = signal_len.max(kernel_len);
                (full_len - out_len) / 2
            }
            ConvMode::Valid => kernel_len - 1,
        }
    }

    /// Calculate 2D output shape
    pub fn output_shape_2d(
        &self,
        signal_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> (usize, usize) {
        (
            self.output_len(signal_shape.0, kernel_shape.0),
            self.output_len(signal_shape.1, kernel_shape.1),
        )
    }
}

// ============================================================================
// Signal Processing Trait
// ============================================================================

/// Algorithmic contract for signal processing operations
///
/// All backends implementing signal processing MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait SignalProcessingAlgorithms<R: Runtime>: FftAlgorithms<R> {
    /// 1D convolution using FFT
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal tensor of shape [..., N]
    /// * `kernel` - Convolution kernel tensor of shape [M] (1D only)
    /// * `mode` - Output mode (Full, Same, Valid)
    ///
    /// # Returns
    ///
    /// Convolved signal with shape [..., output_len] where output_len depends on mode.
    fn convolve(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;

    /// 2D convolution using FFT
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal tensor of shape [..., H, W]
    /// * `kernel` - Convolution kernel tensor of shape [Kh, Kw] (2D only)
    /// * `mode` - Output mode (Full, Same, Valid)
    fn convolve2d(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;

    /// 1D cross-correlation
    ///
    /// Cross-correlation is related to convolution by:
    /// `correlate(x, y) = convolve(x, reverse(y))`
    fn correlate(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;

    /// 2D cross-correlation
    fn correlate2d(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;

    /// Short-Time Fourier Transform
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

    /// Inverse Short-Time Fourier Transform
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

    /// Compute power spectrogram from signal
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

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate signal dtype for convolution (must be F32 or F64)
pub fn validate_signal_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate that kernel is 1D
pub fn validate_kernel_1d(kernel: &[usize], op: &'static str) -> Result<()> {
    if kernel.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "kernel",
            reason: format!("{op} requires 1D kernel, got {}-D", kernel.len()),
        });
    }
    Ok(())
}

/// Validate that kernel is 2D
pub fn validate_kernel_2d(kernel: &[usize], op: &'static str) -> Result<()> {
    if kernel.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "kernel",
            reason: format!("{op} requires 2D kernel, got {}-D", kernel.len()),
        });
    }
    Ok(())
}

/// Validate STFT parameters
pub fn validate_stft_params(n_fft: usize, hop_length: usize, op: &'static str) -> Result<()> {
    if n_fft == 0 || !n_fft.is_power_of_two() {
        return Err(Error::InvalidArgument {
            arg: "n_fft",
            reason: format!("{op} requires n_fft to be a positive power of 2, got {n_fft}"),
        });
    }
    if hop_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "hop_length",
            reason: format!("{op} requires hop_length > 0, got {hop_length}"),
        });
    }
    Ok(())
}

/// Calculate next power of 2 >= n
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// Calculate number of STFT frames
pub fn stft_num_frames(signal_len: usize, n_fft: usize, hop_length: usize, center: bool) -> usize {
    let padded_len = if center {
        signal_len + n_fft
    } else {
        signal_len
    };

    if padded_len < n_fft {
        0
    } else {
        (padded_len - n_fft) / hop_length + 1
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_mode_output_len() {
        assert_eq!(ConvMode::Full.output_len(10, 3), 12);
        assert_eq!(ConvMode::Same.output_len(10, 3), 10);
        assert_eq!(ConvMode::Valid.output_len(10, 3), 8);

        // kernel longer than signal
        assert_eq!(ConvMode::Same.output_len(3, 10), 10);
    }

    #[test]
    fn test_conv_mode_slice_start() {
        assert_eq!(ConvMode::Full.slice_start(10, 3), 0);
        assert_eq!(ConvMode::Same.slice_start(10, 3), 1);
        assert_eq!(ConvMode::Valid.slice_start(10, 3), 2);
    }

    #[test]
    fn test_stft_num_frames() {
        let frames = stft_num_frames(1000, 256, 64, true);
        let expected = (1000 + 256 - 256) / 64 + 1;
        assert_eq!(frames, expected);
    }
}
