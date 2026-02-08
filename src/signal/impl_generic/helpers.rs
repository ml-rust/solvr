//! Generic helper functions for signal processing.
//!
//! These functions implement tensor operations using numr's tensor API,
//! keeping all data on device without CPU roundtrips.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Reverse 1D tensor using tensor.flip() - no CPU roundtrip.
///
/// Note: `_client` is unused here since flip() is a tensor method, but kept
/// in signature for consistency with other impl_generic functions.
pub fn reverse_1d_impl<R, C>(_client: &C, tensor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if tensor.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_1d requires 1D tensor".to_string(),
        });
    }

    // flip() returns a view with reversed strides - no data copy needed
    tensor.flip(0)
}

/// Reverse 2D tensor - flip both dimensions using tensor.flip_dims().
///
/// Note: `_client` is unused here since flip_dims() is a tensor method, but kept
/// in signature for consistency with other impl_generic functions.
pub fn reverse_2d_impl<R, C>(_client: &C, tensor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    if tensor.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_2d requires 2D tensor".to_string(),
        });
    }

    // flip_dims reverses along both dimensions - no data copy needed
    tensor.flip_dims(&[0, 1])
}

/// Element-wise complex multiplication using numr's mul() which handles complex types.
///
/// For complex tensors, numr's mul() performs proper complex multiplication:
/// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
pub fn complex_mul_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let dtype = a.dtype();

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "complex_mul",
        });
    }

    // numr's mul() handles complex multiplication properly
    client.mul(a, b)
}

/// Element-wise complex division: a / b for complex tensors.
///
/// Computes (a_re + a_im*i) / (b_re + b_im*i) using the formula:
/// result_re = (a_re*b_re + a_im*b_im) / (b_re^2 + b_im^2)
/// result_im = (a_im*b_re - a_re*b_im) / (b_re^2 + b_im^2)
#[allow(dead_code)]
pub fn complex_divide_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ComplexOps<R> + RuntimeClient<R>,
{
    let dtype = a.dtype();

    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "complex_divide",
        });
    }

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    // For complex division: a/b = a * conj(b) / |b|^2
    // conj(b) = b_re - b_im*i
    // |b|^2 = b_re^2 + b_im^2

    // Get conjugate of b
    let b_conj = client.conj(b)?;

    // Compute a * conj(b) (complex multiplication)
    let numerator = client.mul(a, &b_conj)?;

    // Compute |b|^2 = b_re^2 + b_im^2
    let b_re = client.real(b)?;
    let b_im = client.imag(b)?;
    let b_re_sq = client.mul(&b_re, &b_re)?;
    let b_im_sq = client.mul(&b_im, &b_im)?;
    let denom = client.add(&b_re_sq, &b_im_sq)?;

    // Divide complex numerator by real denominator
    client.complex_div_real(&numerator, &denom)
}

/// Detrend a tensor by removing mean (constant) or linear trend.
///
/// This operates on the last dimension of the tensor, allowing batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetrendMode {
    /// No detrending
    #[default]
    None,
    /// Remove mean (constant detrend)
    Constant,
    /// Remove linear trend
    Linear,
}

/// Detrend a 1D or 2D tensor along the last dimension.
///
/// For batched inputs (2D), detrends each row independently.
#[allow(dead_code)]
pub fn detrend_tensor_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    mode: DetrendMode,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    match mode {
        DetrendMode::None => Ok(tensor.clone()),
        DetrendMode::Constant => {
            // Remove mean along last dimension
            let ndim = tensor.ndim();
            let last_dim = ndim - 1;

            // Compute mean along last dimension, keeping dims for broadcasting
            let mean = client.mean(tensor, &[last_dim], true)?;

            // Subtract mean
            client.sub(tensor, &mean)
        }
        DetrendMode::Linear => {
            // Linear detrend: y - (a + b*x) where a and b are least squares fit
            let ndim = tensor.ndim();
            let last_dim = ndim - 1;
            let n = tensor.shape()[last_dim];
            let _device = tensor.device();
            let dtype = tensor.dtype();

            if n < 2 {
                return Ok(tensor.clone());
            }

            // Create x indices: [0, 1, 2, ..., n-1]
            let x = client.arange(0.0, n as f64, 1.0, dtype)?;

            // Compute means
            let x_mean = (n - 1) as f64 / 2.0;
            let y_mean = client.mean(tensor, &[last_dim], true)?;

            // Center the data
            let y_centered = client.sub(tensor, &y_mean)?;
            let x_centered = client.add_scalar(&x, -x_mean)?;

            // For linear regression: b = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
            // Broadcast x_centered to match tensor shape for multiplication

            // Compute numerator: sum of (x - x_mean) * (y - y_mean) along last dim
            let x_centered_broadcast = if ndim == 1 {
                x_centered.clone()
            } else {
                // Reshape x_centered to broadcast with tensor
                let mut shape = vec![1usize; ndim];
                shape[last_dim] = n;
                x_centered.reshape(&shape)?
            };

            let xy_product = client.mul(&x_centered_broadcast, &y_centered)?;
            let numerator = client.sum(&xy_product, &[last_dim], true)?;

            // Compute denominator: sum of (x - x_mean)^2
            // This is a constant for a given n, computed analytically:
            // sum_{i=0}^{n-1} (i - (n-1)/2)^2 = n(n^2-1)/12
            let denom_val = (n as f64) * ((n * n - 1) as f64) / 12.0;

            // Compute slope b
            let b = client.div_scalar(&numerator, denom_val)?;

            // Compute intercept a = y_mean - b * x_mean
            let b_x_mean = client.mul_scalar(&b, x_mean)?;
            let a = client.sub(&y_mean, &b_x_mean)?;

            // Compute trend: a + b * x
            let trend_bx = client.mul(&b, &x_centered_broadcast)?;
            let trend = client.add(&a, &trend_bx)?;

            // Subtract trend from original
            client.sub(tensor, &trend)
        }
    }
}

/// Compute |complex|^power for spectrogram using tensor operations.
///
/// Computes (re^2 + im^2)^(power/2) for complex tensors.
pub fn complex_magnitude_pow_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    power: f64,
    output_dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let dtype = tensor.dtype();

    // Validate input is complex
    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "complex_magnitude_pow",
        });
    }

    // Extract real and imaginary parts - stays on device
    let re = client.real(tensor)?;
    let im = client.imag(tensor)?;

    // Compute magnitude squared: re^2 + im^2
    let re_sq = client.mul(&re, &re)?;
    let im_sq = client.mul(&im, &im)?;
    let mag_sq = client.add(&re_sq, &im_sq)?;

    // Apply power
    let result = if (power - 2.0).abs() < 1e-10 {
        // power = 2: just return magnitude squared
        mag_sq
    } else if (power - 1.0).abs() < 1e-10 {
        // power = 1: return magnitude = sqrt(mag_sq)
        client.sqrt(&mag_sq)?
    } else {
        // General case: mag_sq^(power/2)
        let half_power = power / 2.0;
        client.pow_scalar(&mag_sq, half_power)?
    };

    // Cast to output dtype if needed
    if result.dtype() != output_dtype {
        client.cast(&result, output_dtype)
    } else {
        Ok(result)
    }
}

/// Extract overlapping segments from a 1D signal as a 2D tensor.
///
/// Returns a tensor of shape [num_segments, nperseg] where each row is a segment.
pub fn extract_segments_impl<R, C>(
    client: &C,
    signal: &Tensor<R>,
    nperseg: usize,
    noverlap: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ShapeOps<R> + RuntimeClient<R>,
{
    if signal.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "signal",
            reason: "Signal must be 1D".to_string(),
        });
    }

    let n = signal.shape()[0];
    let step = nperseg - noverlap;

    if step == 0 {
        return Err(Error::InvalidArgument {
            arg: "noverlap",
            reason: "noverlap must be less than nperseg".to_string(),
        });
    }

    let num_segments = if n >= nperseg {
        (n - nperseg) / step + 1
    } else {
        0
    };

    if num_segments == 0 {
        return Err(Error::InvalidArgument {
            arg: "signal",
            reason: "Signal too short for given segment parameters".to_string(),
        });
    }

    // Extract each segment using narrow and stack them
    let mut segments: Vec<Tensor<R>> = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let start = i * step;
        let segment = signal.narrow(0, start, nperseg)?;
        segments.push(segment);
    }

    // Stack into [num_segments, nperseg]
    let segment_refs: Vec<&Tensor<R>> = segments.iter().collect();
    client.stack(&segment_refs, 0)
}

/// Compute power spectrum from complex FFT result: |FFT|^2.
///
/// Uses conj(fft) * fft to get the power spectrum, extracting the real part.
pub fn power_spectrum_impl<R, C>(client: &C, fft_result: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ComplexOps<R> + RuntimeClient<R>,
{
    // Power = conj(fft) * fft = |fft|^2
    let conj = client.conj(fft_result)?;
    let power_complex = client.mul(&conj, fft_result)?;

    // Extract real part (imaginary should be ~0)
    client.real(&power_complex)
}
