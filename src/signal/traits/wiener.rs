//! Wiener filter algorithm traits.
//!
//! Provides Wiener filtering for noise reduction in signals and images.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Wiener filter algorithms.
///
/// All backends implementing Wiener filtering MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait WienerAlgorithms<R: Runtime> {
    /// Apply a 1D Wiener filter for noise reduction.
    ///
    /// The Wiener filter is an optimal linear filter that minimizes the
    /// mean square error between the estimated and true signal. It works
    /// in the frequency domain using:
    ///
    /// ```text
    /// H(f) = P_s(f) / (P_s(f) + P_n(f))
    /// ```
    ///
    /// where P_s is signal power and P_n is noise power.
    ///
    /// # Arguments
    ///
    /// * `x` - Input noisy signal (1D tensor)
    /// * `kernel_size` - Size of the local window for power estimation (must be odd, default: 3)
    /// * `noise` - Noise power (if None, estimated from the signal)
    ///
    /// # Returns
    ///
    /// Filtered signal with reduced noise.
    ///
    /// # Algorithm
    ///
    /// 1. Estimate local signal variance using a sliding window
    /// 2. Estimate noise variance (given or from minimum local variance)
    /// 3. Compute Wiener filter: (local_var - noise) / local_var, clamped to [0, 1]
    /// 4. Apply filter: output = mean + filter * (input - mean)
    fn wiener(
        &self,
        x: &Tensor<R>,
        kernel_size: Option<usize>,
        noise: Option<f64>,
    ) -> Result<Tensor<R>>;

    /// Apply a 2D Wiener filter for noise reduction in images.
    ///
    /// # Arguments
    ///
    /// * `x` - Input noisy image (2D tensor)
    /// * `kernel_size` - Size of the local window (height, width), both must be odd
    /// * `noise` - Noise variance (if None, estimated from the image)
    ///
    /// # Returns
    ///
    /// Filtered image with reduced noise.
    fn wiener2d(
        &self,
        x: &Tensor<R>,
        kernel_size: Option<(usize, usize)>,
        noise: Option<f64>,
    ) -> Result<Tensor<R>>;
}
