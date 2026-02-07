//! Generic helper functions for statistics implementations.
//!
//! These helpers work with any numr Runtime (CPU, CUDA, WebGPU).

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Extract a scalar f64 from a 0-D or 1-element tensor.
///
/// Works with any Runtime backend.
pub fn extract_scalar<R: Runtime>(t: &Tensor<R>) -> Result<f64> {
    if t.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: format!("expected scalar (1 element), got {} elements", t.numel()),
        });
    }

    let t = t.contiguous();
    match t.dtype() {
        DType::F32 => {
            let data: Vec<f32> = t.to_vec();
            Ok(data[0] as f64)
        }
        DType::F64 => {
            let data: Vec<f64> = t.to_vec();
            Ok(data[0])
        }
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "extract_scalar",
        }),
    }
}

/// Compute ranks for Spearman correlation (works on any Runtime).
///
/// Uses scatter to assign ranks on-device — no GPU↔CPU transfers.
pub fn compute_ranks<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let x_contig = x.contiguous();
    let n = x_contig.numel();
    let device = client.device();

    // argsort gives indices that would sort x
    let indices = client.argsort(&x_contig, 0, false)?;

    // ranks = 1-based: arange(1, n+1)
    let ranks_seq = client.arange(1.0, (n + 1) as f64, 1.0, x.dtype())?;

    // Scatter ranks into original positions:
    // output[indices[i]] = ranks_seq[i] → ranks_seq[i] = i+1
    let zeros = Tensor::<R>::full_scalar(&[n], x.dtype(), 0.0, device);
    client.scatter(&zeros, 0, &indices, &ranks_seq)
}

/// Compute median of a 1-D tensor on-device (single scalar transfer at end).
pub fn tensor_median_scalar<R, C>(client: &C, x: &Tensor<R>) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let sorted = client.sort(x, 0, false)?;
    let n = sorted.numel();
    if n % 2 == 1 {
        extract_scalar(&sorted.narrow(0, n / 2, 1)?)
    } else {
        let pair = sorted.narrow(0, n / 2 - 1, 2)?;
        let all_dims: Vec<usize> = (0..pair.ndim()).collect();
        extract_scalar(&client.mean(&pair, &all_dims, false)?)
    }
}
