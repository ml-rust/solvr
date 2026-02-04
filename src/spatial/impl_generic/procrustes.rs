//! Generic Procrustes analysis implementation.
//!
//! Kabsch algorithm for finding optimal rotation between point sets.

use crate::spatial::impl_generic::rotation::rotation_from_matrix_impl;
use crate::spatial::traits::procrustes::ProcrustesResult;
use crate::spatial::{validate_matching_dims, validate_points_2d, validate_points_dtype};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute Procrustes analysis using the Kabsch algorithm.
///
/// Finds optimal rotation, translation, and optional scaling to align
/// source points to target points.
pub fn procrustes_impl<R, C>(
    client: &C,
    source: &Tensor<R>,
    target: &Tensor<R>,
    scaling: bool,
    reflection: bool,
) -> Result<ProcrustesResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    validate_points_dtype(source.dtype(), "procrustes")?;
    validate_points_dtype(target.dtype(), "procrustes")?;
    validate_points_2d(source.shape(), "procrustes")?;
    validate_points_2d(target.shape(), "procrustes")?;
    validate_matching_dims(source.shape(), target.shape(), "procrustes")?;

    let _n = source.shape()[0];
    let d = source.shape()[1];

    if source.shape() != target.shape() {
        return Err(Error::InvalidArgument {
            arg: "source/target",
            reason: format!(
                "Source and target must have same shape. Got {:?} and {:?}",
                source.shape(),
                target.shape()
            ),
        });
    }

    let device = source.device();
    let _dtype = source.dtype();

    // 1. Center both point sets
    let source_mean = client.mean(source, &[0], true)?;
    let target_mean = client.mean(target, &[0], true)?;

    let source_centered = client.sub(source, &source_mean.broadcast_to(source.shape())?)?;
    let target_centered = client.sub(target, &target_mean.broadcast_to(target.shape())?)?;

    // 2. Compute cross-covariance matrix H = source_centered.T @ target_centered
    let source_t = source_centered.transpose(0, 1)?;
    let h = client.matmul(&source_t, &target_centered)?;

    // 3. SVD of H
    let svd = client.svd_decompose(&h)?;
    let u = svd.u;
    let s = svd.s;
    let vt = svd.vt;

    // 4. Optimal rotation R = V @ U.T
    let v = vt.transpose(0, 1)?;
    let ut = u.transpose(0, 1)?;
    let mut r = client.matmul(&v, &ut)?;

    // 5. Handle reflection if needed
    let det = LinearAlgebraAlgorithms::det(client, &r)?;
    let det_val: Vec<f64> = det.to_vec();

    if det_val[0] < 0.0 && !reflection {
        // Flip sign of last column of V
        let v_data: Vec<f64> = v.to_vec();
        let mut v_corrected = v_data.clone();

        // Flip last column
        for i in 0..d {
            v_corrected[i * d + (d - 1)] = -v_corrected[i * d + (d - 1)];
        }

        let v_new = Tensor::<R>::from_slice(&v_corrected, &[d, d], device);
        r = client.matmul(&v_new, &ut)?;
    }

    // Convert rotation matrix to Rotation struct
    let rotation = rotation_from_matrix_impl(client, &r)?;

    // 6. Compute scale if requested
    let scale = if scaling {
        // scale = trace(S) / ||source_centered||^2
        let s_data: Vec<f64> = s.to_vec();
        let trace_s: f64 = s_data.iter().sum();

        let source_sq = client.mul(&source_centered, &source_centered)?;
        let source_norm_sq = client.sum(&source_sq, &[0, 1], false)?;
        let source_norm_sq_val: Vec<f64> = source_norm_sq.to_vec();

        trace_s / source_norm_sq_val[0]
    } else {
        1.0
    };

    // 7. Compute translation: t = target_centroid - scale * R @ source_centroid
    let source_mean_flat = source_mean.reshape(&[d])?;
    let rotated_mean = client.matmul(&r, &source_mean_flat.reshape(&[d, 1])?)?;
    let rotated_mean = rotated_mean.reshape(&[d])?;
    let scaled_rotated_mean = client.mul_scalar(&rotated_mean, scale)?;
    let target_mean_flat = target_mean.reshape(&[d])?;
    let translation = client.sub(&target_mean_flat, &scaled_rotated_mean)?;

    // 8. Transform source points
    // transformed = scale * (source @ R.T) + translation
    let r_t = r.transpose(0, 1)?;
    let rotated = client.matmul(source, &r_t)?;
    let scaled = client.mul_scalar(&rotated, scale)?;
    let transformed = client.add(&scaled, &translation.broadcast_to(scaled.shape())?)?;

    // 9. Compute disparity
    let diff = client.sub(&transformed, target)?;
    let diff_sq = client.mul(&diff, &diff)?;
    let disparity_tensor = client.sum(&diff_sq, &[0, 1], false)?;
    let disparity_val: Vec<f64> = disparity_tensor.to_vec();
    let disparity = disparity_val[0];

    Ok(ProcrustesResult {
        rotation,
        translation,
        scale,
        transformed,
        disparity,
    })
}

/// Orthogonal Procrustes: find orthogonal matrix R minimizing ||A @ R - B||_F.
pub fn orthogonal_procrustes_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<(Tensor<R>, f64)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument {
            arg: "a/b",
            reason: format!(
                "A and B must have same shape. Got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    // SVD of A.T @ B
    let at = a.transpose(0, 1)?;
    let m = client.matmul(&at, b)?;
    let svd = client.svd_decompose(&m)?;
    let u = svd.u;
    let vt = svd.vt;

    // R = V @ U.T
    let v = vt.transpose(0, 1)?;
    let ut = u.transpose(0, 1)?;
    let r = client.matmul(&v, &ut)?;

    // Compute residual ||A @ R - B||_F
    let ar = client.matmul(a, &r)?;
    let diff = client.sub(&ar, b)?;
    let diff_sq = client.mul(&diff, &diff)?;
    let residual_tensor = client.sum(&diff_sq, &[0, 1], false)?;
    let residual_val: Vec<f64> = residual_tensor.to_vec();
    let residual = residual_val[0].sqrt();

    Ok((r, residual))
}
