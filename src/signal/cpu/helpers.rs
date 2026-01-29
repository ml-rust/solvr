//! CPU helper functions for signal processing.
//!
//! This module contains utility functions for complex arithmetic, tensor reversal,
//! and magnitude computation. These are internal implementations used by the
//! CPU signal processing backend.

use crate::signal::stft_core::{
    complex_mul_c128, complex_mul_c64, magnitude_pow_f32, magnitude_pow_f64,
    reverse_1d_into, reverse_2d_into,
};
use numr::dtype::{Complex128, Complex64, DType};
use numr::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Reverse 1D tensor.
///
/// Creates a new tensor with elements in reversed order.
pub(crate) fn reverse_1d(tensor: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_1d requires 1D tensor".to_string(),
        });
    }

    let len = tensor_contig.shape()[0];
    let output = Tensor::<CpuRuntime>::empty(&[len], dtype, tensor_contig.storage().device());

    match dtype {
        DType::F32 => {
            // SAFETY: We verified the tensor is contiguous and 1D with `len` elements.
            // The output tensor is freshly allocated with the same size.
            // Both pointers are valid for `len` elements of the respective type.
            let src = unsafe {
                std::slice::from_raw_parts(tensor_contig.storage().ptr() as *const f32, len)
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(output.storage().ptr() as *mut f32, len)
            };
            reverse_1d_into(src, dst);
        }
        DType::F64 => {
            // SAFETY: Same as F32 case above.
            let src = unsafe {
                std::slice::from_raw_parts(tensor_contig.storage().ptr() as *const f64, len)
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(output.storage().ptr() as *mut f64, len)
            };
            reverse_1d_into(src, dst);
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "reverse_1d",
            })
        }
    }

    Ok(output)
}

/// Reverse 2D tensor (flip both dimensions).
///
/// Creates a new tensor with both rows and columns reversed.
pub(crate) fn reverse_2d(tensor: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_2d requires 2D tensor".to_string(),
        });
    }

    let h = tensor_contig.shape()[0];
    let w = tensor_contig.shape()[1];
    let output = Tensor::<CpuRuntime>::empty(&[h, w], dtype, tensor_contig.storage().device());

    match dtype {
        DType::F32 => {
            // SAFETY: We verified the tensor is contiguous and 2D with h*w elements.
            // The output tensor is freshly allocated with the same dimensions.
            let src = unsafe {
                std::slice::from_raw_parts(tensor_contig.storage().ptr() as *const f32, h * w)
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(output.storage().ptr() as *mut f32, h * w)
            };
            reverse_2d_into(src, dst, h, w);
        }
        DType::F64 => {
            // SAFETY: Same as F32 case above.
            let src = unsafe {
                std::slice::from_raw_parts(tensor_contig.storage().ptr() as *const f64, h * w)
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(output.storage().ptr() as *mut f64, h * w)
            };
            reverse_2d_into(src, dst, h, w);
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "reverse_2d",
            })
        }
    }

    Ok(output)
}

/// Element-wise complex multiplication.
///
/// Computes (a * b) element-wise for complex tensors.
pub(crate) fn complex_mul(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
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

    let a_contig = a.contiguous();
    let b_contig = b.contiguous();
    let output = Tensor::<CpuRuntime>::empty(a.shape(), dtype, a_contig.storage().device());

    let numel = a.numel();

    match dtype {
        DType::Complex64 => {
            let a_ptr = a_contig.storage().ptr() as *const Complex64;
            let b_ptr = b_contig.storage().ptr() as *const Complex64;
            let out_ptr = output.storage().ptr() as *mut Complex64;

            for i in 0..numel {
                // SAFETY: We verified both tensors have the same shape with `numel` elements.
                // The output is allocated with the same size. All accesses are within bounds.
                unsafe {
                    let av = *a_ptr.add(i);
                    let bv = *b_ptr.add(i);
                    *out_ptr.add(i) = complex_mul_c64(av, bv);
                }
            }
        }
        DType::Complex128 => {
            let a_ptr = a_contig.storage().ptr() as *const Complex128;
            let b_ptr = b_contig.storage().ptr() as *const Complex128;
            let out_ptr = output.storage().ptr() as *mut Complex128;

            for i in 0..numel {
                // SAFETY: Same as Complex64 case above.
                unsafe {
                    let av = *a_ptr.add(i);
                    let bv = *b_ptr.add(i);
                    *out_ptr.add(i) = complex_mul_c128(av, bv);
                }
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "complex_mul",
            })
        }
    }

    Ok(output)
}

/// Compute |complex|^power for spectrogram.
///
/// Computes the magnitude of each complex element raised to the given power.
pub(crate) fn complex_magnitude_pow(
    tensor: &Tensor<CpuRuntime>,
    power: f64,
    output_dtype: DType,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();
    let numel = tensor.numel();

    let output =
        Tensor::<CpuRuntime>::empty(tensor.shape(), output_dtype, tensor_contig.storage().device());

    match (dtype, output_dtype) {
        (DType::Complex64, DType::F32) => {
            let src = tensor_contig.storage().ptr() as *const Complex64;
            let dst = output.storage().ptr() as *mut f32;

            for i in 0..numel {
                // SAFETY: We're iterating over `numel` elements. Both source and destination
                // tensors have exactly `numel` elements. Pointer arithmetic is within bounds.
                unsafe {
                    let c = *src.add(i);
                    *dst.add(i) = magnitude_pow_f32(c, power);
                }
            }
        }
        (DType::Complex128, DType::F64) => {
            let src = tensor_contig.storage().ptr() as *const Complex128;
            let dst = output.storage().ptr() as *mut f64;

            for i in 0..numel {
                // SAFETY: Same as above.
                unsafe {
                    let c = *src.add(i);
                    *dst.add(i) = magnitude_pow_f64(c, power);
                }
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "complex_magnitude_pow",
            })
        }
    }

    Ok(output)
}
