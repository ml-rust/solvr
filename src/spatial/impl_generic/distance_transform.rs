//! Generic distance transform implementations.
//!
//! EDT uses the Felzenszwalb & Huttenlocher separable parabola envelope algorithm.
//! CDT uses iterative pad+narrow+minimum propagation (fully on-device).
use crate::DType;

use crate::spatial::traits::distance_transform::DistanceTransformMetric;
use numr::error::{Error, Result};
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic distance transform implementation.
pub fn distance_transform_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    metric: DistanceTransformMetric,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    match metric {
        DistanceTransformMetric::Euclidean => distance_transform_edt_impl(client, input),
        DistanceTransformMetric::CityBlock => chamfer_distance_impl(client, input, false),
        DistanceTransformMetric::Chessboard => chamfer_distance_impl(client, input, true),
    }
}

/// Exact Euclidean Distance Transform using Felzenszwalb & Huttenlocher.
///
/// NOTE: This algorithm is inherently sequential per-row (parabola envelope construction).
/// There is no equivalent tensor-op formulation. GPU alternatives (Jump Flooding Algorithm)
/// require complex coordinate tracking and would be a separate implementation. The `to_vec()`
/// and `from_slice()` here are at the API boundary (input → sequential transform → output),
/// which is acceptable per solvr's transfer policy.
pub fn distance_transform_edt_impl<R, C>(_client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "distance_transform_edt requires at least 1D input".to_string(),
        });
    }

    let shape = input.shape().to_vec();
    let device = input.device();

    // NOTE: CPU transfer required — Felzenszwalb parabola envelope is inherently sequential.
    let data: Vec<f64> = input.to_vec();
    let inf = 1e18;
    let mut dist_sq: Vec<f64> = data
        .iter()
        .map(|&v| if v != 0.0 { 0.0 } else { inf })
        .collect();

    // Apply 1D EDT along each dimension
    for dim in 0..ndim {
        let n = shape[dim];
        let stride: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();

        for outer_idx in 0..outer {
            for inner_idx in 0..stride {
                // Extract 1D slice
                let mut f = vec![0.0f64; n];
                for (i, f_val) in f.iter_mut().enumerate() {
                    let flat = outer_idx * (n * stride) + i * stride + inner_idx;
                    *f_val = dist_sq[flat];
                }

                // 1D EDT (squared distances)
                let dt = edt_1d_squared(&f);

                // Write back
                for (i, &dt_val) in dt.iter().enumerate() {
                    let flat = outer_idx * (n * stride) + i * stride + inner_idx;
                    dist_sq[flat] = dt_val;
                }
            }
        }
    }

    // Take square root
    let result: Vec<f64> = dist_sq.iter().map(|&d| d.sqrt()).collect();
    Tensor::from_slice(&result, &shape, device)
}

/// 1D squared distance transform using parabola envelope (Felzenszwalb & Huttenlocher).
fn edt_1d_squared(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n == 0 {
        return vec![];
    }

    let mut d = vec![0.0f64; n];
    let mut v = vec![0usize; n]; // locations of parabolas
    let mut z = vec![0.0f64; n + 1]; // boundaries between parabolas
    let mut k = 0usize; // number of parabolas in lower envelope

    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for q in 1..n {
        loop {
            let vk = v[k] as f64;
            let qq = q as f64;
            let s = ((f[q] + qq * qq) - (f[v[k]] + vk * vk)) / (2.0 * qq - 2.0 * vk);

            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for (q, d_val) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let diff = q as f64 - v[k] as f64;
        *d_val = diff * diff + f[v[k]];
    }

    d
}

/// Pad a single axis of a tensor with a constant value.
///
/// Uses numr's `pad` which expects PyTorch-style padding: `[last_before, last_after, ...]`.
fn pad_single_axis<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    axis: usize,
    before: usize,
    after: usize,
    value: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R> + RuntimeClient<R>,
{
    let ndim = tensor.ndim();
    // PyTorch pad format: pairs from last dim backward
    // For axis `a` in an ndim tensor, position in padding array = (ndim - 1 - a) * 2
    let mut padding = vec![0usize; ndim * 2];
    let pad_idx = (ndim - 1 - axis) * 2;
    padding[pad_idx] = before;
    padding[pad_idx + 1] = after;
    client.pad(tensor, &padding, value)
}

/// Chamfer distance transform using iterative pad+narrow+minimum propagation.
///
/// For each axis, pads with a large value, takes shifted views (left/right neighbors),
/// adds 1, and takes element-wise minimum. Iterates until convergence.
/// Fully on-device — no GPU↔CPU transfers except single scalar convergence check.
fn chamfer_distance_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    _chessboard: bool,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "chamfer distance requires at least 1D input".to_string(),
        });
    }

    let shape = input.shape().to_vec();
    let total: usize = shape.iter().product();
    let large = (total + 1) as f64;

    // Build foreground mask: nonzero => true
    let zero_tensor = Tensor::from_slice(&vec![0.0; total], &shape, input.device())?;
    let fg_mask = client.ne(input, &zero_tensor)?;
    let zero = Tensor::from_slice(&vec![0.0; total], &shape, input.device())?;
    let large_tensor = Tensor::from_slice(&vec![large; total], &shape, input.device())?;

    // Initialize: foreground = 0, background = large
    let mut dist = client.where_cond(&fg_mask, &zero, &large_tensor)?;

    // Iterate until convergence. For city block distance on an N-D grid,
    // worst case needs diameter passes, but typically converges in a few.
    let max_iter = total; // absolute upper bound
    for _ in 0..max_iter {
        let prev = dist.clone();

        for (axis, &axis_len) in shape.iter().enumerate() {
            if axis_len <= 1 {
                continue;
            }

            // Pad this axis with `large` on both sides
            let padded = pad_single_axis(client, &dist, axis, 1, 1, large)?;

            // Left neighbor (shifted view) and right neighbor
            let left = padded.narrow(axis as isize, 0, axis_len)?;
            let right = padded.narrow(axis as isize, 2, axis_len)?;

            // neighbor + 1
            let left_plus1 = client.add_scalar(&left, 1.0)?;
            let right_plus1 = client.add_scalar(&right, 1.0)?;

            // Take minimum of current, left+1, right+1
            dist = client.minimum(&dist, &left_plus1)?;
            dist = client.minimum(&dist, &right_plus1)?;
        }

        // Convergence check — single scalar transfer (acceptable)
        let diff = client.sub(&dist, &prev)?;
        let diff_abs = client.abs(&diff)?;
        let diff_sum = client.sum(&diff_abs, &[], false)?;
        let val: Vec<f64> = diff_sum.to_vec();
        if val[0] < 0.5 {
            break;
        }
    }

    Ok(dist)
}
