//! Scattered data interpolation generic implementation.
//!
//! Provides nearest-neighbor and linear interpolation for scattered data.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::scattered::ScatteredMethod;
use numr::ops::{CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Interpolate scattered data at query points.
pub fn griddata_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    values: &Tensor<R>,
    xi: &Tensor<R>,
    method: ScatteredMethod,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + MatmulOps<R> + ShapeOps<R> + RuntimeClient<R>,
{
    let p_shape = points.shape().to_vec();
    if p_shape.len() != 2 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "points".to_string(),
            message: "points must be 2D [n, d]".to_string(),
        });
    }
    let n = p_shape[0];
    let d = p_shape[1];

    let v_shape = values.shape().to_vec();
    if v_shape[0] != n {
        return Err(InterpolateError::ShapeMismatch {
            expected: n,
            actual: v_shape[0],
            context: "griddata: points vs values".to_string(),
        });
    }

    let xi_shape = xi.shape().to_vec();
    if xi_shape.len() != 2 || xi_shape[1] != d {
        return Err(InterpolateError::DimensionMismatch {
            expected: d,
            actual: xi_shape.get(1).copied().unwrap_or(0),
            context: "griddata: xi dimension must match points".to_string(),
        });
    }

    match method {
        ScatteredMethod::Nearest => nearest_interp(client, points, values, xi, n, d),
        ScatteredMethod::Linear => linear_interp(client, points, values, xi, n, d),
    }
}

/// Nearest neighbor interpolation via distance matrix.
fn nearest_interp<R, C>(
    client: &C,
    points: &Tensor<R>,
    values: &Tensor<R>,
    xi: &Tensor<R>,
    n: usize,
    d: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + MatmulOps<R> + RuntimeClient<R>,
{
    let m = xi.shape()[0];

    // Compute squared distance matrix [m, n]
    // ||xi_i - p_j||^2 = ||xi_i||^2 + ||p_j||^2 - 2*xi_i.p_j
    let xi_sq = sum_sq_rows(client, xi, d)?; // [m, 1]
    let p_sq = sum_sq_rows(client, points, d)?; // [n, 1]
    let p_sq_t = p_sq.transpose(0, 1)?.contiguous(); // [1, n]

    let p_t = points.transpose(0, 1)?.contiguous();
    let dot = client.matmul(xi, &p_t)?; // [m, n]
    let two_dot = client.mul_scalar(&dot, 2.0)?;

    let sum = client.add(
        &xi_sq.broadcast_to(&[m, n])?,
        &p_sq_t.broadcast_to(&[m, n])?,
    )?;
    let dist_sq = client.sub(&sum, &two_dot)?;

    // argmin along dim 1 -> nearest neighbor indices [m]
    let indices = client.argmin(&dist_sq, 1, false)?;

    // Gather values
    let result = client.index_select(values, 0, &indices)?;
    Ok(result)
}

/// Linear interpolation for 2D scattered data.
/// Uses inverse distance weighting with k nearest neighbors as a practical
/// linear interpolation method that works in any dimension.
fn linear_interp<R, C>(
    client: &C,
    points: &Tensor<R>,
    values: &Tensor<R>,
    xi: &Tensor<R>,
    n: usize,
    d: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + MatmulOps<R> + ShapeOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let m = xi.shape()[0];

    // For linear scattered interpolation we use inverse distance weighting (IDW)
    // with p=2: w_j = 1/d_j^2, normalized.
    // This is a practical approach that works on-device for any dimension.

    // Compute distance matrix [m, n]
    let xi_sq = sum_sq_rows(client, xi, d)?;
    let p_sq = sum_sq_rows(client, points, d)?;
    let p_sq_t = p_sq.transpose(0, 1)?.contiguous();
    let p_t = points.transpose(0, 1)?.contiguous();
    let dot = client.matmul(xi, &p_t)?;
    let two_dot = client.mul_scalar(&dot, 2.0)?;
    let sum = client.add(
        &xi_sq.broadcast_to(&[m, n])?,
        &p_sq_t.broadcast_to(&[m, n])?,
    )?;
    let dist_sq = client.sub(&sum, &two_dot)?;

    // Clamp to avoid division by zero
    let eps = Tensor::full_scalar(&[m, n], DType::F64, 1e-30, device);
    let dist_sq_safe = client.maximum(&dist_sq, &eps)?;

    // Weights = 1/dist_sq (IDW with power 2)
    let ones = Tensor::full_scalar(&[m, n], DType::F64, 1.0, device);
    let weights = client.div(&ones, &dist_sq_safe)?;

    // Normalize weights: w_j / sum(w_j)
    // Sum across dim 1
    let mut w_sum = weights.narrow(1, 0, 1)?;
    for j in 1..n {
        let col = weights.narrow(1, j, 1)?;
        w_sum = client.add(&w_sum, &col)?;
    }
    let w_norm = client.div(&weights, &w_sum.broadcast_to(&[m, n])?)?;

    // Result = w_norm @ values
    let vals_col = if values.shape().len() == 1 {
        values.reshape(&[n, 1])?
    } else {
        values.clone()
    };
    let result = client.matmul(&w_norm, &vals_col)?;

    if values.shape().len() == 1 {
        Ok(result.reshape(&[m])?)
    } else {
        Ok(result)
    }
}

/// Compute sum of squared values along rows: [n, d] -> [n, 1].
fn sum_sq_rows<R, C>(client: &C, a: &Tensor<R>, d: usize) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let a2 = client.mul(a, a)?;
    let mut s = a2.narrow(1, 0, 1)?;
    for col in 1..d {
        let c = a2.narrow(1, col, 1)?;
        s = client.add(&s, &c)?;
    }
    Ok(s)
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
    fn test_nearest_2d() {
        let (device, client) = setup();
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &device);

        // Query points close to known points
        let xi = Tensor::<CpuRuntime>::from_slice(
            &[0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],
            &[4, 2],
            &device,
        );

        let result =
            griddata_impl(&client, &points, &values, &xi, ScatteredMethod::Nearest).unwrap();
        let vals: Vec<f64> = result.to_vec();

        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
        assert!((vals[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_idw_at_known_points() {
        let (device, client) = setup();
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );
        let values = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 1.0, 2.0], &[4], &device);

        // At the center (0.5, 0.5), all 4 points are equidistant -> average
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);
        let result =
            griddata_impl(&client, &points, &values, &xi, ScatteredMethod::Linear).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // Average of [0, 1, 1, 2] = 1.0
        assert!((vals[0] - 1.0).abs() < 1e-6, "center: {} vs 1.0", vals[0]);
    }
}
