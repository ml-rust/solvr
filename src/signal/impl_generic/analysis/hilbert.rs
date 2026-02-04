//! Hilbert transform for computing analytic signal.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use crate::signal::traits::analysis::HilbertResult;
use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute the analytic signal using the Hilbert transform.
///
/// Uses numr FFT operations (backend-optimized).
///
/// The analytic signal is:
/// ```text
/// x_a(t) = x(t) + j * H{x(t)}
/// ```
/// where H{x(t)} is the Hilbert transform of x(t).
pub fn hilbert_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<HilbertResult<R>>
where
    R: Runtime,
    C: FftAlgorithms<R>
        + ComplexOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + ShapeOps<R>
        + RuntimeClient<R>,
{
    let n = x.shape()[0];
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    // Convert real to complex (imaginary part = 0)
    let zeros = Tensor::zeros(x.shape(), x.dtype(), device);
    let x_complex = client.make_complex(x, &zeros)?;

    // Compute FFT via numr
    let fft = client.fft(&x_complex, FftDirection::Forward, FftNormalization::None)?;

    // Construct Hilbert frequency response as a tensor
    // H[0] = 1 (DC)
    // H[1:N/2] = 2 (positive frequencies)
    // H[N/2] = 1 (Nyquist, if N is even)
    // H[N/2+1:] = 0 (negative frequencies)
    let mut h_data = vec![0.0f64; n];
    h_data[0] = 1.0; // DC

    let half = n / 2;
    for hi in h_data.iter_mut().take(half).skip(1) {
        *hi = 2.0; // Positive frequencies
    }

    if n.is_multiple_of(2) {
        h_data[half] = 1.0; // Nyquist for even N
    } else {
        h_data[half] = 2.0; // Include in positive for odd N
    }
    // Negative frequencies (h_data[half+1..]) stay at 0

    let h = Tensor::from_slice(&h_data, &[n], device);

    // Multiply FFT by H
    // Need to handle Complex * Real: extract real/imag, multiply, recombine
    let fft_re = client.real(&fft)?;
    let fft_im = client.imag(&fft)?;
    let scaled_re = client.mul(&fft_re, &h)?;
    let scaled_im = client.mul(&fft_im, &h)?;
    let scaled_fft = client.make_complex(&scaled_re, &scaled_im)?;

    // Compute IFFT via numr
    let analytic = client.fft(
        &scaled_fft,
        FftDirection::Inverse,
        FftNormalization::Backward,
    )?;

    // Extract real and imaginary parts
    let real = client.real(&analytic)?;
    let imag = client.imag(&analytic)?;

    Ok(HilbertResult { real, imag })
}
