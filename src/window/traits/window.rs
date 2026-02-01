use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Trait for generating window functions used in signal processing.
///
/// Window functions are multiplied with a signal before FFT analysis to reduce
/// spectral leakage. Different windows offer different tradeoffs between
/// frequency resolution and sidelobe suppression.
///
/// # Backend Implementation
///
/// All backends use CPU-based generation since window functions are typically
/// small arrays. The result is transferred to the target device if necessary.
///
/// # Example
///
/// ```ignore
/// use solvr::window::WindowFunctions;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
/// use numr::dtype::DType;
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let window = client.hann_window(1024, DType::F32, &device)?;
/// ```
pub trait WindowFunctions<R: Runtime> {
    /// Generate a Hann (Hanning) window.
    ///
    /// The Hann window is a general-purpose window with good frequency resolution
    /// and moderate sidelobe suppression (-31.5 dB first sidelobe).
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
    /// ```
    fn hann_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Hamming window.
    ///
    /// The Hamming window has better sidelobe suppression than Hann (-42.7 dB first
    /// sidelobe) but doesn't go to zero at the edges.
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.54 - 0.46 * cos(2*pi*n / N)
    /// ```
    fn hamming_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Blackman window.
    ///
    /// The Blackman window has excellent sidelobe suppression (-58 dB first sidelobe)
    /// at the cost of wider main lobe than Hann/Hamming.
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.42 - 0.5 * cos(2*pi*n / N) + 0.08 * cos(4*pi*n / N)
    /// ```
    fn blackman_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Kaiser window.
    ///
    /// The Kaiser window is a flexible window with adjustable parameter beta that
    /// controls the tradeoff between main lobe width and sidelobe level.
    ///
    /// # Formula
    ///
    /// ```text
    /// w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
    /// ```
    ///
    /// # Beta Parameter Guidelines
    ///
    /// | Beta  | Sidelobe Attenuation | Approximate Equivalent |
    /// |-------|---------------------|------------------------|
    /// | 0     | -13 dB             | Rectangular            |
    /// | 5     | -50 dB             | Hamming                |
    /// | 6     | -60 dB             | Hann                   |
    /// | 8.6   | -90 dB             | Blackman               |
    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Tensor<R>>;
}

/// Validate window dtype (must be F32 or F64).
pub fn validate_window_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate window size (must be positive).
pub fn validate_window_size(size: usize, op: &'static str) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!("{op} requires size > 0"),
        });
    }
    Ok(())
}
