//! Local extrema detection algorithm traits.
//!
//! Provides algorithms for finding local minima and maxima in signals.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Mode for handling boundaries when finding extrema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtremumMode {
    /// Wrap around at boundaries (periodic signal).
    Wrap,
    /// Clip comparisons at boundaries (no wraparound).
    #[default]
    Clip,
}

/// Result from extrema detection.
#[derive(Debug, Clone)]
pub struct ExtremaResult<R: Runtime> {
    /// Indices of detected extrema.
    pub indices: Vec<usize>,
    /// Values at extrema locations.
    pub values: Tensor<R>,
}

/// Local extrema detection algorithms.
///
/// All backends implementing extrema detection MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait ExtremaAlgorithms<R: Runtime> {
    /// Find local minima in a 1D signal.
    ///
    /// A point is a local minimum if it's smaller than all points within
    /// `order` samples on either side.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal (1D tensor)
    /// * `order` - Number of samples to compare on each side (default: 1)
    /// * `mode` - How to handle boundaries (Wrap or Clip)
    ///
    /// # Returns
    ///
    /// [`ExtremaResult`] containing indices and values of local minima.
    ///
    /// # Example
    ///
    /// For order=1, a point x[i] is a local minimum if:
    /// - x[i] < x[i-1] AND x[i] < x[i+1]
    ///
    /// For order=2, a point x[i] is a local minimum if:
    /// - x[i] < x[i-2] AND x[i] < x[i-1] AND x[i] < x[i+1] AND x[i] < x[i+2]
    fn argrelmin(
        &self,
        x: &Tensor<R>,
        order: usize,
        mode: ExtremumMode,
    ) -> Result<ExtremaResult<R>>;

    /// Find local maxima in a 1D signal.
    ///
    /// A point is a local maximum if it's larger than all points within
    /// `order` samples on either side.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal (1D tensor)
    /// * `order` - Number of samples to compare on each side (default: 1)
    /// * `mode` - How to handle boundaries (Wrap or Clip)
    ///
    /// # Returns
    ///
    /// [`ExtremaResult`] containing indices and values of local maxima.
    fn argrelmax(
        &self,
        x: &Tensor<R>,
        order: usize,
        mode: ExtremumMode,
    ) -> Result<ExtremaResult<R>>;

    /// Find both local minima and maxima in a 1D signal.
    ///
    /// Convenience function that calls both `argrelmin` and `argrelmax`.
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal (1D tensor)
    /// * `order` - Number of samples to compare on each side (default: 1)
    /// * `mode` - How to handle boundaries (Wrap or Clip)
    ///
    /// # Returns
    ///
    /// Tuple of (minima, maxima) as [`ExtremaResult`]s.
    fn argrelextrema(
        &self,
        x: &Tensor<R>,
        order: usize,
        mode: ExtremumMode,
    ) -> Result<(ExtremaResult<R>, ExtremaResult<R>)> {
        let minima = self.argrelmin(x, order, mode)?;
        let maxima = self.argrelmax(x, order, mode)?;
        Ok((minima, maxima))
    }
}
