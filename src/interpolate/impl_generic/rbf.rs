//! Radial Basis Function interpolation generic implementation.
//!
//! Constructs the RBF kernel matrix from pairwise distances, optionally
//! augments with polynomial terms, solves the system, and evaluates at
//! query points.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::rbf::{RbfKernel, RbfModel};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fit an RBF interpolant.
pub fn rbf_fit_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    values: &Tensor<R>,
    kernel: RbfKernel,
    epsilon: Option<f64>,
    smoothing: f64,
) -> InterpolateResult<RbfModel<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + MatmulOps<R>
        + ShapeOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
{
    let device = client.device();
    let shape = points.shape().to_vec();
    if shape.len() != 2 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "points".to_string(),
            message: "points must be 2D [n, d]".to_string(),
        });
    }
    let n = shape[0];
    let d = shape[1];

    // Auto-select epsilon if not provided: mean nearest-neighbor distance
    let eps = match epsilon {
        Some(e) => e,
        None => auto_epsilon(client, points, n)?,
    };

    // Compute pairwise distance matrix [n, n]
    let dist = cdist_euclidean(client, points, points)?;

    // Apply kernel function
    let mut kernel_mat = apply_kernel(client, &dist, kernel, eps)?;

    // Add smoothing to diagonal
    if smoothing > 0.0 {
        let eye = Tensor::from_slice(
            &(0..n)
                .flat_map(|i| (0..n).map(move |j| if i == j { smoothing } else { 0.0 }))
                .collect::<Vec<_>>(),
            &[n, n],
            device,
        );
        kernel_mat = client.add(&kernel_mat, &eye)?;
    }

    // Determine polynomial degree based on kernel
    let poly_degree = match kernel {
        RbfKernel::ThinPlateSpline | RbfKernel::Linear | RbfKernel::Cubic | RbfKernel::Quintic => {
            1 // linear polynomial augmentation
        }
        _ => 0, // no augmentation needed for positive definite kernels
    };

    if poly_degree == 0 {
        // Simple system: K * w = v
        let vals_col = if values.shape().len() == 1 {
            values.reshape(&[n, 1])?
        } else {
            values.clone()
        };

        let weights_col =
            LinearAlgebraAlgorithms::solve(client, &kernel_mat, &vals_col).map_err(|e| {
                InterpolateError::NumericalError {
                    message: format!("RBF solve failed: {}", e),
                }
            })?;
        let weights = if values.shape().len() == 1 {
            weights_col.reshape(&[n])?
        } else {
            weights_col
        };

        Ok(RbfModel {
            centers: points.clone(),
            weights,
            kernel,
            epsilon: eps,
            poly_coeffs: None,
            dim: d,
        })
    } else {
        // Augmented system with polynomial terms:
        // [ K   P ] [ w ]   [ v ]
        // [ P^T 0 ] [ c ] = [ 0 ]
        let p_cols = 1 + d; // constant + linear terms
        // Build polynomial matrix P [n, 1+d]: [1, x1, x2, ..., xd]
        let ones_col = Tensor::full_scalar(&[n, 1], DType::F64, 1.0, device);
        let p_mat = client.cat(&[&ones_col, points], -1)?;

        // Build augmented system
        let zeros_pp = Tensor::zeros(&[p_cols, p_cols], DType::F64, device);

        // Top row: [K, P]
        let kernel_mat = kernel_mat.contiguous();
        let p_mat = p_mat.contiguous();
        let top = client.cat(&[&kernel_mat, &p_mat], 1)?;
        // Bottom row: [P^T, 0]
        let p_t = p_mat.transpose(0, 1)?.contiguous();
        let bottom = client.cat(&[&p_t, &zeros_pp], 1)?;
        // Full: [[K, P], [P^T, 0]]
        let aug_mat = client.cat(&[&top, &bottom], 0)?;

        // Augmented rhs
        let vals_col = if values.shape().len() == 1 {
            values.reshape(&[n, 1])?
        } else {
            values.clone()
        };
        let n_out = vals_col.shape().get(1).copied().unwrap_or(1);
        let zeros_rhs = Tensor::zeros(&[p_cols, n_out], DType::F64, device);
        let aug_rhs = client.cat(&[&vals_col, &zeros_rhs], 0)?;

        let aug_mat = aug_mat.contiguous();
        let aug_rhs = aug_rhs.contiguous();
        let solution = LinearAlgebraAlgorithms::solve(client, &aug_mat, &aug_rhs).map_err(|e| {
            InterpolateError::NumericalError {
                message: format!("RBF augmented solve failed: {}", e),
            }
        })?;

        // Split solution into weights and polynomial coefficients
        let weights = solution.narrow(0, 0, n)?.contiguous();
        let poly_coeffs = solution.narrow(0, n, p_cols)?.contiguous();

        let weights = if values.shape().len() == 1 {
            weights.reshape(&[n])?
        } else {
            weights
        };
        let poly_coeffs = if values.shape().len() == 1 {
            poly_coeffs.reshape(&[p_cols])?
        } else {
            poly_coeffs
        };

        Ok(RbfModel {
            centers: points.clone(),
            weights,
            kernel,
            epsilon: eps,
            poly_coeffs: Some(poly_coeffs),
            dim: d,
        })
    }
}

/// Evaluate an RBF model at query points.
pub fn rbf_evaluate_impl<R, C>(
    client: &C,
    model: &RbfModel<R>,
    query: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + MatmulOps<R>
        + ShapeOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
{
    let device = client.device();
    let q_shape = query.shape().to_vec();
    let m = q_shape[0];

    // Compute distances from query points to centers
    let dist = cdist_euclidean(client, query, &model.centers)?;

    // Apply kernel
    let kernel_vals = apply_kernel(client, &dist, model.kernel, model.epsilon)?;

    // Compute weighted sum: result = kernel_vals @ weights
    let weights = if model.weights.shape().len() == 1 {
        model.weights.reshape(&[model.weights.shape()[0], 1])?
    } else {
        model.weights.clone()
    };
    let mut result = client.matmul(&kernel_vals, &weights)?;

    // Add polynomial contribution if present
    if let Some(ref poly_coeffs) = model.poly_coeffs {
        let ones_col: Tensor<R> = Tensor::full_scalar(&[m, 1], DType::F64, 1.0, device);
        let p_query = client.cat(&[&ones_col, query], -1)?;
        let pc = if poly_coeffs.shape().len() == 1 {
            poly_coeffs.reshape(&[poly_coeffs.shape()[0], 1])?
        } else {
            poly_coeffs.clone()
        };
        let poly_term = client.matmul(&p_query, &pc)?;
        result = client.add(&result, &poly_term)?;
    }

    // Squeeze if single output
    if result.shape().len() == 2 && result.shape()[1] == 1 {
        result = result.reshape(&[m])?;
    }

    Ok(result)
}

/// Compute Euclidean distance matrix between two point sets.
/// a: [n, d], b: [m, d] -> result: [n, m]
fn cdist_euclidean<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + MatmulOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let d = a.shape()[1];

    // ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 * a_i . b_j
    // a_sq: [n, 1], b_sq: [1, m]
    let a_sq = {
        let a2 = client.mul(a, a)?;
        // Sum over dim 1
        let mut s = a2.narrow(1, 0, 1)?;
        for col in 1..d {
            let c = a2.narrow(1, col, 1)?;
            s = client.add(&s, &c)?;
        }
        s // [n, 1]
    };

    let b_sq = {
        let b2 = client.mul(b, b)?;
        let mut s = b2.narrow(1, 0, 1)?;
        for col in 1..d {
            let c = b2.narrow(1, col, 1)?;
            s = client.add(&s, &c)?;
        }
        s.transpose(0, 1)?.contiguous() // [1, m]
    };

    let b_t = b.transpose(0, 1)?.contiguous();
    let ab = client.matmul(a, &b_t)?; // [n, m]
    let two_ab = client.mul_scalar(&ab, 2.0)?;

    // Broadcast a_sq [n,1] + b_sq [1,m] -> [n,m]
    let a_sq_b = a_sq.broadcast_to(&[n, m])?.contiguous();
    let b_sq_b = b_sq.broadcast_to(&[n, m])?.contiguous();
    let sum_sq = client.add(&a_sq_b, &b_sq_b)?;
    let dist_sq = client.sub(&sum_sq, &two_ab)?;

    // Clamp to non-negative before sqrt (numerical safety)
    let zero = Tensor::zeros(&[n, m], DType::F64, device);
    let dist_sq_safe = client.maximum(&dist_sq, &zero)?;
    let dist = client.sqrt(&dist_sq_safe)?;

    Ok(dist)
}

/// Apply RBF kernel function element-wise to distance matrix.
fn apply_kernel<R, C>(
    client: &C,
    dist: &Tensor<R>,
    kernel: RbfKernel,
    epsilon: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let shape = dist.shape().to_vec();

    match kernel {
        RbfKernel::ThinPlateSpline => {
            // r^2 * ln(r), with 0*ln(0) = 0
            let r2 = client.mul(dist, dist)?;
            let ln_r = client.log(dist)?;
            let result = client.mul(&r2, &ln_r)?;
            // Replace NaN (from 0*ln(0)) with 0
            let zero = Tensor::zeros(&shape, DType::F64, device);
            let is_zero = client.eq(dist, &zero)?;
            Ok(client.where_cond(&is_zero, &zero, &result)?)
        }
        RbfKernel::Multiquadric => {
            // sqrt(1 + (r/eps)^2)
            let r_eps = client.mul_scalar(dist, 1.0 / epsilon)?;
            let r_eps2 = client.mul(&r_eps, &r_eps)?;
            let one_plus = client.add_scalar(&r_eps2, 1.0)?;
            Ok(client.sqrt(&one_plus)?)
        }
        RbfKernel::InverseMultiquadric => {
            // 1/sqrt(1 + (r/eps)^2)
            let r_eps = client.mul_scalar(dist, 1.0 / epsilon)?;
            let r_eps2 = client.mul(&r_eps, &r_eps)?;
            let one_plus = client.add_scalar(&r_eps2, 1.0)?;
            let sq = client.sqrt(&one_plus)?;
            let one = Tensor::full_scalar(&shape, DType::F64, 1.0, device);
            Ok(client.div(&one, &sq)?)
        }
        RbfKernel::Gaussian => {
            // exp(-(r/eps)^2)
            let r_eps = client.mul_scalar(dist, 1.0 / epsilon)?;
            let r_eps2 = client.mul(&r_eps, &r_eps)?;
            let neg = client.mul_scalar(&r_eps2, -1.0)?;
            Ok(client.exp(&neg)?)
        }
        RbfKernel::Linear => {
            // r
            Ok(dist.clone())
        }
        RbfKernel::Cubic => {
            // r^3
            let r2 = client.mul(dist, dist)?;
            Ok(client.mul(&r2, dist)?)
        }
        RbfKernel::Quintic => {
            // r^5
            let r2 = client.mul(dist, dist)?;
            let r4 = client.mul(&r2, &r2)?;
            Ok(client.mul(&r4, dist)?)
        }
    }
}

/// Auto-select epsilon as mean nearest-neighbor distance (fully on-device).
fn auto_epsilon<R, C>(client: &C, points: &Tensor<R>, n: usize) -> InterpolateResult<f64>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + MatmulOps<R> + RuntimeClient<R>,
{
    if n < 2 {
        return Ok(1.0);
    }

    // Compute pairwise distance matrix [n, n] on-device
    let dist = cdist_euclidean(client, points, points)?;

    // Add large value to diagonal so self-distance isn't the minimum
    let big = Tensor::full_scalar(&[n], DType::F64, 1e30, client.device());
    let big_diag = client.diagflat(&big)?;
    let dist_masked = client.add(&dist, &big_diag)?;

    // Min along rows → [n] nearest-neighbor distances
    let nn_dists = client.min(&dist_masked, &[1], false)?;

    // Mean → scalar
    let mean_dist = client.mean(&nn_dists, &[0], false)?;

    // Single scalar extraction (acceptable: returns config parameter)
    Ok(mean_dist.item::<f64>().unwrap_or(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_rbf_gaussian_1d() {
        let (device, client) = setup();
        // 1D points as [n, 1]
        let points = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4, 1], &device);
        let values = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);

        let model = rbf_fit_impl(
            &client,
            &points,
            &values,
            RbfKernel::Gaussian,
            Some(1.0),
            0.0,
        )
        .unwrap();

        // Evaluate at known points — should recover values
        let result = rbf_evaluate_impl(&client, &model, &points).unwrap();
        let vals: Vec<f64> = result.to_vec();
        let expected = [0.0, 1.0, 4.0, 9.0];

        for i in 0..4 {
            assert!(
                (vals[i] - expected[i]).abs() < 1e-6,
                "point {}: {} vs {}",
                i,
                vals[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_rbf_thin_plate_2d() {
        let (device, client) = setup();
        // 2D points
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );
        let values = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 1.0, 2.0], &[4], &device);

        let model = rbf_fit_impl(
            &client,
            &points,
            &values,
            RbfKernel::ThinPlateSpline,
            None,
            0.0,
        )
        .unwrap();

        // Evaluate at known points
        let result = rbf_evaluate_impl(&client, &model, &points).unwrap();
        let vals: Vec<f64> = result.to_vec();
        let expected = [0.0, 1.0, 1.0, 2.0];

        for i in 0..4 {
            assert!(
                (vals[i] - expected[i]).abs() < 1e-6,
                "point {}: {} vs {}",
                i,
                vals[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_rbf_multiquadric() {
        let (device, client) = setup();
        let points = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4, 1], &device);
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 1.0, 0.0], &[4], &device);

        let model = rbf_fit_impl(
            &client,
            &points,
            &values,
            RbfKernel::Multiquadric,
            Some(1.0),
            0.0,
        )
        .unwrap();

        let result = rbf_evaluate_impl(&client, &model, &points).unwrap();
        let vals: Vec<f64> = result.to_vec();
        let expected = [1.0, 0.0, 1.0, 0.0];

        for i in 0..4 {
            assert!(
                (vals[i] - expected[i]).abs() < 1e-6,
                "point {}: {} vs {}",
                i,
                vals[i],
                expected[i]
            );
        }
    }
}
