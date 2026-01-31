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
pub fn compute_ranks<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let x_contig = x.contiguous();
    let n = x_contig.numel();

    // Get sorted indices
    let indices = client.argsort(&x_contig, 0, false)?;

    // Create ranks (1-based)
    let mut ranks = vec![0.0f64; n];
    let indices_data: Vec<i64> = indices.to_vec();

    for (rank, &idx) in indices_data.iter().enumerate() {
        ranks[idx as usize] = (rank + 1) as f64;
    }

    Ok(Tensor::<R>::from_slice(
        &ranks,
        x_contig.shape(),
        client.device(),
    ))
}
