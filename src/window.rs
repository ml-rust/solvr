//! Window function algorithms for signal processing
//!
//! This module defines the [`WindowFunctions`] trait that provides window functions
//! commonly used in spectral analysis and signal processing. All window functions
//! use the periodic formulation (suitable for FFT-based analysis).
//!
//! # Supported Windows
//!
//! - **Hann (Hanning)**: General-purpose window with good frequency resolution
//! - **Hamming**: Similar to Hann but with better sidelobe suppression
//! - **Blackman**: Excellent sidelobe suppression at the cost of main lobe width
//! - **Kaiser**: Parametric window with adjustable frequency resolution vs sidelobe tradeoff
//!
//! # Window Function Comparison
//!
//! | Window     | First Sidelobe | Sidelobe Rolloff | Main Lobe Width | Best For |
//! |------------|---------------|------------------|-----------------|----------|
//! | Rectangular| -13 dB        | -6 dB/octave     | Narrowest       | Transient analysis |
//! | Hann       | -31.5 dB      | -18 dB/octave    | Moderate        | General purpose |
//! | Hamming    | -42.7 dB      | -6 dB/octave     | Moderate        | Audio processing |
//! | Blackman   | -58 dB        | -18 dB/octave    | Wide            | High dynamic range |
//! | Kaiser     | Adjustable    | Adjustable       | Adjustable      | Custom requirements |
//!
//! # Choosing a Window
//!
//! - **Hann**: Start here. Best all-around choice for most applications.
//! - **Hamming**: Use when you need consistent sidelobe attenuation (-42 dB floor).
//! - **Blackman**: Use when spectral leakage must be minimized (e.g., detecting weak signals).
//! - **Kaiser**: Use when you need precise control over the resolution/leakage tradeoff.
//!
//! ## Kaiser Beta Guidelines
//!
//! The Kaiser window's `beta` parameter controls the tradeoff:
//!
//! | Beta | Approximate Sidelobe | Equivalent Window |
//! |------|---------------------|-------------------|
//! | 0    | -13 dB              | Rectangular       |
//! | 5    | -50 dB              | Hamming           |
//! | 6    | -60 dB              | Hann              |
//! | 8.6  | -90 dB              | Blackman          |
//! | 14   | -120 dB             | (very narrow)     |
//!
//! # Implementation Notes
//!
//! Window functions are implemented on CPU regardless of the target device, as they are
//! typically small arrays where GPU acceleration provides no benefit. The generated
//! window is transferred to the target device when needed.
//!
//! # Mathematical Definitions (Periodic Formulation)
//!
//! For a window of size N with n = 0, 1, ..., N-1:
//!
//! ```text
//! Hann:     w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
//! Hamming:  w[n] = 0.54 - 0.46 * cos(2*pi*n / N)
//! Blackman: w[n] = 0.42 - 0.5 * cos(2*pi*n / N) + 0.08 * cos(4*pi*n / N)
//! Kaiser:   w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
//! ```
//!
//! Where I0 is the modified Bessel function of the first kind, order 0.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::f64::consts::PI;

// ============================================================================
// Window Functions Trait
// ============================================================================

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

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate window dtype (must be F32 or F64)
pub fn validate_window_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate window size (must be positive)
pub fn validate_window_size(size: usize, op: &'static str) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!("{op} requires size > 0"),
        });
    }
    Ok(())
}

// ============================================================================
// Window Generation Helpers (CPU-based)
// ============================================================================

/// Generate Hann window values as f64
pub fn generate_hann_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

/// Generate Hamming window values as f64
pub fn generate_hamming_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.54 - 0.46 * x.cos()
        })
        .collect()
}

/// Generate Blackman window values as f64
pub fn generate_blackman_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect()
}

/// Generate Kaiser window values as f64
pub fn generate_kaiser_f64(size: usize, beta: f64) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }

    let n = size as f64;
    let half_n = (n - 1.0) / 2.0;
    let i0_beta = bessel_i0(beta);

    (0..size)
        .map(|i| {
            let x = (i as f64 - half_n) / half_n;
            let arg = beta * (1.0 - x * x).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0.
///
/// Uses the polynomial approximation from Abramowitz & Stegun.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492
                        + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        let exp_ax = ax.exp();
        (exp_ax / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537
                                        + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

// ============================================================================
// Backend Implementations
// ============================================================================

// CPU Implementation
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl WindowFunctions<CpuRuntime> for CpuClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "hann_window")?;
        validate_window_dtype(dtype, "hann_window")?;
        let values = generate_hann_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "hamming_window")?;
        validate_window_dtype(dtype, "hamming_window")?;
        let values = generate_hamming_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "blackman_window")?;
        validate_window_dtype(dtype, "blackman_window")?;
        let values = generate_blackman_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        validate_window_dtype(dtype, "kaiser_window")?;
        let values = generate_kaiser_f64(size, beta);
        create_window_tensor(values, dtype, device)
    }
}

fn create_window_tensor(
    values: Vec<f64>,
    dtype: DType,
    device: &<CpuRuntime as Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let size = values.len();
    match dtype {
        DType::F32 => {
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            Ok(Tensor::<CpuRuntime>::from_slice(&values_f32, &[size], device))
        }
        DType::F64 => Ok(Tensor::<CpuRuntime>::from_slice(&values, &[size], device)),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "window",
        }),
    }
}

// CUDA Implementation
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

#[cfg(feature = "cuda")]
impl WindowFunctions<CudaRuntime> for CudaClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "hann_window")?;
        validate_window_dtype(dtype, "hann_window")?;
        let values = generate_hann_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "hamming_window")?;
        validate_window_dtype(dtype, "hamming_window")?;
        let values = generate_hamming_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "blackman_window")?;
        validate_window_dtype(dtype, "blackman_window")?;
        let values = generate_blackman_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        validate_window_dtype(dtype, "kaiser_window")?;
        let values = generate_kaiser_f64(size, beta);
        create_window_tensor_cuda(values, dtype, device)
    }
}

#[cfg(feature = "cuda")]
fn create_window_tensor_cuda(
    values: Vec<f64>,
    dtype: DType,
    device: &CudaDevice,
) -> Result<Tensor<CudaRuntime>> {
    let size = values.len();
    match dtype {
        DType::F32 => {
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            Ok(Tensor::<CudaRuntime>::from_slice(&values_f32, &[size], device))
        }
        DType::F64 => Ok(Tensor::<CudaRuntime>::from_slice(&values, &[size], device)),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "window",
        }),
    }
}

// WebGPU Implementation
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

#[cfg(feature = "wgpu")]
impl WindowFunctions<WgpuRuntime> for WgpuClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "hann_window")?;
        // WebGPU only supports F32
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "hann_window (WebGPU only supports F32)",
            });
        }
        let values = generate_hann_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(&values_f32, &[size], device))
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "hamming_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "hamming_window (WebGPU only supports F32)",
            });
        }
        let values = generate_hamming_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(&values_f32, &[size], device))
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "blackman_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "blackman_window (WebGPU only supports F32)",
            });
        }
        let values = generate_blackman_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(&values_f32, &[size], device))
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "kaiser_window (WebGPU only supports F32)",
            });
        }
        let values = generate_kaiser_f64(size, beta);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(&values_f32, &[size], device))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = generate_hann_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0].abs() < 1e-10);
        assert!((window[1] - window[7]).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_window() {
        let window = generate_hamming_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0] > 0.05);
    }

    #[test]
    fn test_blackman_window() {
        let window = generate_blackman_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0].abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_window() {
        let window = generate_kaiser_f64(8, 5.0);
        assert_eq!(window.len(), 8);
        for &w in &window {
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_window_size_edge_cases() {
        assert!(generate_hann_f64(0).is_empty());
        assert_eq!(generate_hann_f64(1), vec![1.0]);
    }
}
