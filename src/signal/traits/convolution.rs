//! Convolution and cross-correlation trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Convolution output mode.
///
/// Determines the size and alignment of the convolution output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvMode {
    /// Full convolution output.
    ///
    /// Output length = N + M - 1, where N = signal length, M = kernel length.
    /// Contains all points where the kernel and signal overlap.
    #[default]
    Full,

    /// Same-size output.
    ///
    /// Output length = max(N, M).
    /// Output is centered relative to the 'full' output.
    /// Matches scipy.signal.convolve behavior.
    Same,

    /// Valid convolution output.
    ///
    /// Output length = max(N, M) - min(N, M) + 1.
    /// Contains only points where the kernel fits entirely within the signal.
    Valid,
}

impl ConvMode {
    /// Calculate output length for 1D convolution.
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

    /// Calculate start offset for slicing full convolution result.
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

    /// Calculate 2D output shape.
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

/// Algorithmic contract for 1D and 2D convolution and cross-correlation operations.
///
/// All backends implementing convolution algorithms MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait ConvolutionAlgorithms<R: Runtime> {
    /// 1D convolution using FFT.
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
    fn convolve(&self, signal: &Tensor<R>, kernel: &Tensor<R>, mode: ConvMode)
    -> Result<Tensor<R>>;

    /// 2D convolution using FFT.
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

    /// 1D cross-correlation.
    ///
    /// Cross-correlation is related to convolution by:
    /// `correlate(x, y) = convolve(x, reverse(y))`
    fn correlate(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;

    /// 2D cross-correlation.
    fn correlate2d(
        &self,
        signal: &Tensor<R>,
        kernel: &Tensor<R>,
        mode: ConvMode,
    ) -> Result<Tensor<R>>;
}

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
}
