//! Median filter algorithm traits.
//!
//! Provides algorithms for median filtering of signals and images.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Median filter algorithms.
///
/// All backends implementing median filtering MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait MedianFilterAlgorithms<R: Runtime> {
    /// Apply a 1D median filter to a signal.
    ///
    /// For each point, replaces it with the median of the surrounding points
    /// within a sliding window.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal (1D tensor)
    /// * `kernel_size` - Size of the filter window (must be odd)
    ///
    /// # Returns
    ///
    /// Filtered signal with same length as input.
    ///
    /// # Edge Handling
    ///
    /// At boundaries, the window is truncated and the median is computed
    /// over available samples only.
    fn medfilt(&self, x: &Tensor<R>, kernel_size: usize) -> Result<Tensor<R>>;

    /// Apply a 2D median filter to an image.
    ///
    /// For each pixel, replaces it with the median of the surrounding pixels
    /// within a sliding 2D window.
    ///
    /// # Arguments
    ///
    /// * `x` - Input image (2D tensor)
    /// * `kernel_size` - Size of the filter window (height, width), both must be odd
    ///
    /// # Returns
    ///
    /// Filtered image with same shape as input.
    ///
    /// # Edge Handling
    ///
    /// At boundaries, the window is truncated and the median is computed
    /// over available samples only.
    fn medfilt2d(&self, x: &Tensor<R>, kernel_size: (usize, usize)) -> Result<Tensor<R>>;
}
