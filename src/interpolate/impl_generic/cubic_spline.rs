//! Cubic spline coefficient computation generic implementation.
//!
//! Computes cubic spline coefficients using dense linear algebra.
//! The tridiagonal system is assembled on-device using diagflat + cat,
//! then solved via `LinearAlgebraAlgorithms::solve`. Zero GPU↔CPU transfers.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::cubic_spline::{SplineBoundary, SplineCoefficients};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::ScalarOps;
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Four diagonal/rhs tensors: (main_diag, upper_diag, lower_diag, rhs).
type DiagonalSystem<R> = (Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>);

/// Compute cubic spline coefficients (fully on-device).
///
/// Builds the tridiagonal system using tensor operations (diagflat, cat)
/// and solves via LU decomposition. All computation stays on device.
pub fn cubic_spline_coefficients<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    boundary: &SplineBoundary,
) -> InterpolateResult<SplineCoefficients<R>>
where
    R: Runtime,
    C: ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n = x.shape()[0];

    if n < 2 {
        return Err(InterpolateError::InsufficientData {
            required: 2,
            actual: n,
            context: "cubic_spline_coefficients".to_string(),
        });
    }

    // h[i] = x[i+1] - x[i], shape [n-1]
    let x_hi = x.narrow(0, 1, n - 1)?;
    let x_lo = x.narrow(0, 0, n - 1)?;
    let h = client.sub(&x_hi, &x_lo)?;

    // slopes[i] = (y[i+1] - y[i]) / h[i], shape [n-1]
    let y_hi = y.narrow(0, 1, n - 1)?;
    let y_lo = y.narrow(0, 0, n - 1)?;
    let dy = client.sub(&y_hi, &y_lo)?;
    let slopes = client.div(&dy, &h)?;

    // Interior diagonals (rows 1..n-2):
    // main[i] = 2*(h[i-1] + h[i])  for i=1..n-2  → computed from h[0:n-2] + h[1:n-1]
    // upper[i] = h[i]              for i=1..n-2  → h[1:n-1]
    // lower[i-1] = h[i-1]          for i=1..n-2  → h[0:n-2]
    // rhs[i] = 3*(slopes[i] - slopes[i-1])
    let h_lo_int = h.narrow(0, 0, n - 2)?.contiguous();
    let h_hi_int = h.narrow(0, 1, n - 2)?.contiguous();
    let main_interior = client.mul_scalar(&client.add(&h_lo_int, &h_hi_int)?, 2.0)?;

    let s_lo = slopes.narrow(0, 0, n - 2)?.contiguous();
    let s_hi = slopes.narrow(0, 1, n - 2)?.contiguous();
    let rhs_interior = client.mul_scalar(&client.sub(&s_hi, &s_lo)?, 3.0)?;

    // Build boundary-dependent diagonals and rhs
    let (main_diag, upper_diag, lower_diag, rhs) = build_boundary_diagonals(
        client,
        &h,
        &slopes,
        &main_interior,
        &rhs_interior,
        boundary,
        n,
    )?;

    // Assemble tridiagonal matrix from diagonals (on-device)
    let a_mat = build_tridiagonal(client, &main_diag, &upper_diag, &lower_diag)?;
    let rhs_col = rhs.reshape(&[n, 1])?;

    // Solve on-device
    let c_col = LinearAlgebraAlgorithms::solve(client, &a_mat, &rhs_col).map_err(|e| {
        InterpolateError::NumericalError {
            message: format!("Failed to solve tridiagonal system: {}", e),
        }
    })?;
    let c_tensor = c_col.reshape(&[n])?;

    // a coefficients = y values
    let a_tensor = y.clone();

    // Compute b and d from c using tensor ops:
    // b[i] = slopes[i] - h[i] * (2*c[i] + c[i+1]) / 3
    // d[i] = (c[i+1] - c[i]) / (3 * h[i])
    let c_left = c_tensor.narrow(0, 0, n - 1)?;
    let c_right = c_tensor.narrow(0, 1, n - 1)?;

    let two_c_left = client.mul_scalar(&c_left, 2.0)?;
    let sum_c = client.add(&two_c_left, &c_right)?;
    let three = Tensor::full_scalar(&[n - 1], DType::F64, 3.0, device);
    let hc_term = client.div(&client.mul(&h, &sum_c)?, &three)?;
    let b_tensor = client.sub(&slopes, &hc_term)?;

    let c_diff = client.sub(&c_right, &c_left)?;
    let three_h = client.mul_scalar(&h, 3.0)?;
    let d_tensor = client.div(&c_diff, &three_h)?;

    Ok((a_tensor, b_tensor, c_tensor, d_tensor))
}

/// Build the boundary-dependent diagonal vectors and rhs (fully on-device).
fn build_boundary_diagonals<R, C>(
    client: &C,
    h: &Tensor<R>,
    slopes: &Tensor<R>,
    main_interior: &Tensor<R>,
    rhs_interior: &Tensor<R>,
    boundary: &SplineBoundary,
    n: usize,
) -> InterpolateResult<DiagonalSystem<R>>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let one = Tensor::full_scalar(&[1], DType::F64, 1.0, device);
    let zero_1 = Tensor::zeros(&[1], DType::F64, device);
    let h_lo_int = h.narrow(0, 0, n - 2)?.contiguous();
    let h_hi_int = h.narrow(0, 1, n - 2)?.contiguous();

    match boundary {
        SplineBoundary::Natural => {
            let main = client.cat(&[&one, main_interior, &one], 0)?;
            let upper = client.cat(&[&zero_1, &h_hi_int], 0)?;
            let lower = client.cat(&[&h_lo_int, &zero_1], 0)?;
            let rhs = client.cat(&[&zero_1, rhs_interior, &zero_1], 0)?;
            Ok((main, upper, lower, rhs))
        }
        SplineBoundary::Clamped { left, right } => {
            let h_first = h.narrow(0, 0, 1)?.contiguous();
            let h_last = h.narrow(0, n - 2, 1)?.contiguous();
            let two_h_first = client.mul_scalar(&h_first, 2.0)?;
            let two_h_last = client.mul_scalar(&h_last, 2.0)?;

            let main = client.cat(&[&two_h_first, main_interior, &two_h_last], 0)?;
            let upper = client.cat(&[&h_first, &h_hi_int], 0)?;
            let lower = client.cat(&[&h_lo_int, &h_last], 0)?;

            let s_first = slopes.narrow(0, 0, 1)?.contiguous();
            let s_last = slopes.narrow(0, n - 2, 1)?.contiguous();
            let left_t = Tensor::full_scalar(&[1], DType::F64, *left, device);
            let right_t = Tensor::full_scalar(&[1], DType::F64, *right, device);
            let rhs_first = client.mul_scalar(&client.sub(&s_first, &left_t)?, 3.0)?;
            let rhs_last = client.mul_scalar(&client.sub(&right_t, &s_last)?, 3.0)?;
            let rhs = client.cat(&[&rhs_first, rhs_interior, &rhs_last], 0)?;

            Ok((main, upper, lower, rhs))
        }
        SplineBoundary::NotAKnot => {
            if n < 4 {
                // Fall back to natural for small n
                let main = client.cat(&[&one, main_interior, &one], 0)?;
                let upper = client.cat(&[&zero_1, &h_hi_int], 0)?;
                let lower = client.cat(&[&h_lo_int, &zero_1], 0)?;
                let rhs = client.cat(&[&zero_1, rhs_interior, &zero_1], 0)?;
                Ok((main, upper, lower, rhs))
            } else {
                // Not-a-knot boundary conditions
                let h0 = h.narrow(0, 0, 1)?.contiguous();
                let h1 = h.narrow(0, 1, 1)?.contiguous();
                let hn3 = h.narrow(0, n - 3, 1)?.contiguous();
                let hn2 = h.narrow(0, n - 2, 1)?.contiguous();

                // Main diagonal: [h1²*h0, interior..., hn3²*hn2]
                let h1_sq = client.mul(&h1, &h1)?;
                let main_first = client.mul(&h1_sq, &h0)?;
                let hn3_sq = client.mul(&hn3, &hn3)?;
                let main_last = client.mul(&hn3_sq, &hn2)?;
                let main = client.cat(&[&main_first, main_interior, &main_last], 0)?;

                // Upper: [-(h0²*h1 + h1²*h0), interior h values...]
                let h0_sq = client.mul(&h0, &h0)?;
                let h0_sq_h1 = client.mul(&h0_sq, &h1)?;
                let h1_sq_h0 = client.mul(&h1_sq, &h0)?;
                let upper_first = client.mul_scalar(&client.add(&h0_sq_h1, &h1_sq_h0)?, -1.0)?;
                let upper = client.cat(&[&upper_first, &h_hi_int], 0)?;

                // Lower: [interior h values..., -(hn2²*hn3 + hn3²*hn2)]
                let hn2_sq = client.mul(&hn2, &hn2)?;
                let hn2_sq_hn3 = client.mul(&hn2_sq, &hn3)?;
                let hn3_sq_hn2 = client.mul(&hn3_sq, &hn2)?;
                let lower_last = client.mul_scalar(&client.add(&hn2_sq_hn3, &hn3_sq_hn2)?, -1.0)?;
                let lower = client.cat(&[&h_lo_int, &lower_last], 0)?;

                // RHS: [h0²*h1*(s1-s0), interior..., hn2²*hn3*(sn2-sn3)]
                let s0 = slopes.narrow(0, 0, 1)?.contiguous();
                let s1 = slopes.narrow(0, 1, 1)?.contiguous();
                let rhs_first = client.mul(&h0_sq_h1, &client.sub(&s1, &s0)?)?;
                let sn3 = slopes.narrow(0, n - 3, 1)?.contiguous();
                let sn2 = slopes.narrow(0, n - 2, 1)?.contiguous();
                let rhs_last = client.mul(&hn2_sq_hn3, &client.sub(&sn2, &sn3)?)?;
                let rhs = client.cat(&[&rhs_first, rhs_interior, &rhs_last], 0)?;

                Ok((main, upper, lower, rhs))
            }
        }
    }
}

/// Build a tridiagonal [n, n] matrix from diagonal vectors (fully on-device).
///
/// Uses diagflat for each diagonal, then cat to shift upper/lower diagonals
/// into position.
fn build_tridiagonal<R, C>(
    client: &C,
    main_diag: &Tensor<R>,
    upper_diag: &Tensor<R>,
    lower_diag: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n = main_diag.shape()[0];

    // Main diagonal → [n, n]
    let main_mat = client.diagflat(main_diag)?;

    // Upper diagonal [n-1] → [n-1, n-1] → shift right: prepend zero column, append zero row
    let upper_small = client.diagflat(upper_diag)?;
    let zero_col = Tensor::zeros(&[n - 1, 1], DType::F64, device);
    let upper_shifted = client.cat(&[&zero_col, &upper_small], 1)?; // [n-1, n]
    let zero_row = Tensor::zeros(&[1, n], DType::F64, device);
    let upper_mat = client.cat(&[&upper_shifted, &zero_row], 0)?; // [n, n]

    // Lower diagonal [n-1] → [n-1, n-1] → shift down: append zero column, prepend zero row
    let lower_small = client.diagflat(lower_diag)?;
    let lower_shifted = client.cat(&[&lower_small, &zero_col], 1)?; // [n-1, n]
    let lower_mat = client.cat(&[&zero_row, &lower_shifted], 0)?; // [n, n]

    let result = client.add(&client.add(&main_mat, &upper_mat)?, &lower_mat)?;
    Ok(result)
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
    fn test_natural_spline_interpolates_knots() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);

        let (a, b, c, d) =
            cubic_spline_coefficients(&client, &x, &y, &SplineBoundary::Natural).unwrap();

        let a_v: Vec<f64> = a.to_vec();
        let b_v: Vec<f64> = b.to_vec();
        let c_v: Vec<f64> = c.to_vec();
        let d_v: Vec<f64> = d.to_vec();

        // At t=1 on segment i: a[i]+b[i]+c[i]+d[i] = y[i+1]
        for i in 0..3 {
            let val = a_v[i] + b_v[i] + c_v[i] + d_v[i];
            assert!(
                (val - a_v[i + 1]).abs() < 1e-10,
                "segment {} endpoint: {} vs {}",
                i,
                val,
                a_v[i + 1]
            );
        }
    }

    #[test]
    fn test_clamped_spline_derivative() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0], &[3], &device);

        let (_a, b, _c, _d) = cubic_spline_coefficients(
            &client,
            &x,
            &y,
            &SplineBoundary::Clamped {
                left: 1.0,
                right: -1.0,
            },
        )
        .unwrap();

        let b_v: Vec<f64> = b.to_vec();
        assert!(
            (b_v[0] - 1.0).abs() < 1e-10,
            "left derivative: {} vs 1.0",
            b_v[0]
        );
    }

    #[test]
    fn test_not_a_knot_spline_continuity() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let (a, b, c, d) =
            cubic_spline_coefficients(&client, &x, &y, &SplineBoundary::NotAKnot).unwrap();

        let a_v: Vec<f64> = a.to_vec();
        let b_v: Vec<f64> = b.to_vec();
        let c_v: Vec<f64> = c.to_vec();
        let d_v: Vec<f64> = d.to_vec();

        for i in 0..4 {
            let val = a_v[i] + b_v[i] + c_v[i] + d_v[i];
            assert!(
                (val - a_v[i + 1]).abs() < 1e-8,
                "segment {} mismatch: {} vs {}",
                i,
                val,
                a_v[i + 1]
            );
        }
    }
}
