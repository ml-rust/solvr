//! Signal resampling using FFT-based method.
//!
//! Uses numr tensor ops - backend-optimized (SIMD on CPU, kernels on GPU).

use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::error::{Error, Result};
use numr::ops::{ComplexOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Resample a signal using FFT-based method.
///
/// Uses numr FFT operations (backend-optimized).
///
/// # Algorithm
///
/// FFT-based resampling works by:
/// 1. Pad input to power-of-2 for FFT efficiency
/// 2. Compute FFT of input signal
/// 3. Manipulate frequencies to achieve target length
/// 4. Compute inverse FFT
///
/// This preserves frequency content while changing the sample rate.
pub fn resample_impl<R, C>(client: &C, x: &Tensor<R>, num: usize, den: usize) -> Result<Tensor<R>>
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

    if num == 0 || den == 0 {
        return Err(Error::InvalidArgument {
            arg: "num/den",
            reason: "Resampling factors must be positive".to_string(),
        });
    }

    // Simple case: no resampling
    if num == den {
        return Ok(x.clone());
    }

    // Compute output length
    let output_len = (n * num).div_ceil(den);

    // FFT requires power-of-2. We'll pad both input and output to common power-of-2.
    let fft_len = n.max(output_len).next_power_of_two();

    // Pad input to fft_len
    let x_padded = if n < fft_len {
        let pad_amount = fft_len - n;
        client.pad(x, &[0, pad_amount], 0.0)?
    } else {
        x.clone()
    };

    // Convert real to complex (imaginary part = 0)
    let zeros = Tensor::zeros(&[fft_len], x.dtype(), device);
    let x_complex = client.make_complex(&x_padded, &zeros)?;

    // Compute FFT via numr
    let fft = client.fft(&x_complex, FftDirection::Forward, FftNormalization::None)?;

    // For resampling, we need to manipulate the FFT to target output_len
    // FFT layout: [DC, f1, f2, ..., f_{N/2-1}, f_{N/2}, -f_{N/2-1}, ..., -f1]
    //
    // Strategy:
    // - For upsampling (output_len > n): zero-pad high frequencies in the middle
    // - For downsampling (output_len < n): truncate high frequencies
    //
    // We take frequencies from the padded FFT (which has fft_len elements).
    // Positive freqs are at indices 0..half_n (roughly)
    // Negative freqs are at indices fft_len - half_n..fft_len

    // How many positive and negative frequencies to keep
    // For a signal of length N:
    // - DC is at index 0
    // - Positive frequencies at indices 1 to N/2 (Nyquist at N/2 for even N)
    // - Negative frequencies at indices N/2+1 to N-1
    //
    // For the output of length M, we need:
    // - Positive frequencies at indices 0 to M/2
    // - Negative frequencies at indices M/2+1 to M-1

    let half_n = n / 2;
    let half_out = output_len / 2;

    // Number of positive frequencies to copy (excluding DC, including Nyquist if even)
    // We copy min(half_n, half_out) positive frequencies
    let pos_copy = half_n.min(half_out);

    // Number of negative frequencies to copy
    // Original has (n - 1) / 2 negative frequencies (indices n/2+1 to n-1)
    // Output needs (output_len - 1) / 2 negative frequencies
    let neg_orig = (n.saturating_sub(1)) / 2;
    let neg_out = (output_len.saturating_sub(1)) / 2;
    let neg_copy = neg_orig.min(neg_out);

    // Build output FFT array
    let out_fft_len = output_len.next_power_of_two();

    // Extract positive frequencies (DC + pos_copy positive frequencies)
    let pos_len = pos_copy + 1; // +1 for DC
    let pos_freqs = fft.narrow(0, 0, pos_len)?;

    // Extract negative frequencies from the end of FFT
    // In the padded FFT, negative frequencies of the original signal are at:
    // fft_len - neg_orig .. fft_len
    // We want neg_copy of them
    let neg_freqs = if neg_copy > 0 {
        Some(fft.narrow(0, fft_len - neg_copy, neg_copy)?)
    } else {
        None
    };

    // Build the target FFT tensor of length output_len
    // Layout: [DC, pos_freqs..., zeros/nyquist..., neg_freqs...]
    //
    // For output_len elements:
    // - Index 0: DC
    // - Indices 1..pos_len: positive frequencies (we have pos_len-1 of these)
    // - Indices output_len - neg_copy..output_len: negative frequencies
    // - Middle: zeros (for upsampling) or nothing (if it fits exactly)

    let new_fft = if let Some(neg) = neg_freqs {
        let middle_len = output_len.saturating_sub(pos_len).saturating_sub(neg_copy);
        if middle_len > 0 {
            // Need zero padding in the middle
            let zeros_real = Tensor::zeros(&[middle_len], x.dtype(), device);
            let zeros_imag = Tensor::zeros(&[middle_len], x.dtype(), device);
            let middle = client.make_complex(&zeros_real, &zeros_imag)?;
            client.cat(&[&pos_freqs, &middle, &neg], 0)?
        } else {
            // Fits exactly or truncated
            if pos_len + neg_copy == output_len {
                client.cat(&[&pos_freqs, &neg], 0)?
            } else if pos_len >= output_len {
                // Very aggressive downsampling - just keep positive frequencies
                pos_freqs.narrow(0, 0, output_len)?
            } else {
                // Truncate negative frequencies
                let neg_keep = output_len - pos_len;
                let neg_trunc = neg.narrow(0, neg_copy - neg_keep, neg_keep)?;
                client.cat(&[&pos_freqs, &neg_trunc], 0)?
            }
        }
    } else {
        // No negative frequencies (n <= 2)
        if output_len > pos_len {
            let pad = output_len - pos_len;
            let zeros_real = Tensor::zeros(&[pad], x.dtype(), device);
            let zeros_imag = Tensor::zeros(&[pad], x.dtype(), device);
            let padding = client.make_complex(&zeros_real, &zeros_imag)?;
            client.cat(&[&pos_freqs, &padding], 0)?
        } else if output_len < pos_len {
            pos_freqs.narrow(0, 0, output_len)?
        } else {
            pos_freqs
        }
    };

    // Verify we have output_len elements
    debug_assert_eq!(new_fft.shape()[0], output_len);

    // Pad to power-of-2 for IFFT if needed
    let curr_len = new_fft.shape()[0];
    let padded_fft = if curr_len < out_fft_len {
        let pad = out_fft_len - curr_len;
        let zeros_real = Tensor::zeros(&[pad], x.dtype(), device);
        let zeros_imag = Tensor::zeros(&[pad], x.dtype(), device);
        let padding = client.make_complex(&zeros_real, &zeros_imag)?;
        client.cat(&[&new_fft, &padding], 0)?
    } else {
        new_fft
    };

    // Scale by ratio to preserve amplitude
    let scale = output_len as f64 / n as f64;
    let fft_re = client.real(&padded_fft)?;
    let fft_im = client.imag(&padded_fft)?;
    let scaled_re = client.mul_scalar(&fft_re, scale)?;
    let scaled_im = client.mul_scalar(&fft_im, scale)?;
    let scaled_fft = client.make_complex(&scaled_re, &scaled_im)?;

    // Compute IFFT via numr
    let result_complex = client.fft(
        &scaled_fft,
        FftDirection::Inverse,
        FftNormalization::Backward,
    )?;

    // Extract real part and truncate to output_len
    let result_real = client.real(&result_complex)?;
    let result = result_real.narrow(0, 0, output_len)?;

    Ok(result)
}
