//! Generic helper functions for signal processing.
//!
//! These functions implement tensor operations using numr's tensor API,
//! keeping all data on device without CPU roundtrips.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
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
