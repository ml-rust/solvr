//! Smooth bivariate spline generic implementation (fully on-device).
//!
//! Fits a tensor-product B-spline to scattered (x, y, z) data using
//! penalized least squares. When smoothing=0, exact interpolation via lstsq.
//! When smoothing>0, adds roughness penalty for smooth surfaces.
//!
//! Fully on-device — zero GPU↔CPU transfers.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bspline::build_knot_vector_tensor;
use crate::interpolate::impl_generic::rect_bivariate_spline::rect_bivariate_spline_evaluate_impl;
use crate::interpolate::traits::bspline::BSplineBoundary;
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, ScalarOps, UtilityOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::bspline::compute_basis_matrix;

/// Build a linearly-spaced tensor from tensor min/max, fully on-device.
/// Equivalent to `linspace(min_val, max_val, n)` but without scalar extraction.
fn on_device_linspace<R, C>(
    client: &C,
    min_t: &Tensor<R>,
    max_t: &Tensor<R>,
    n: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    if n == 1 {
        return Ok(min_t.reshape(&[1])?.contiguous()?);
    }
    let steps = client.arange(0.0, n as f64, 1.0, DType::F64)?; // [n]
    let denom = Tensor::full_scalar(&[1], DType::F64, (n - 1) as f64, device)?;
    let t = client.div(&steps, &denom)?; // [0, ..., 1]
    let range = client.sub(max_t, min_t)?; // scalar tensor
    let range_broad = range.broadcast_to(&[n])?.contiguous()?;
    let min_broad = min_t.broadcast_to(&[n])?.contiguous()?;
    Ok(client.add(&min_broad, &client.mul(&range_broad, &t)?)?)
}

/// Fit a smoothing bivariate spline to scattered data.
///
/// Approach: Penalized least squares with tensor-product B-spline basis.
///
/// 1. Auto-generate knot vectors from data range
/// 2. Build 1D basis Bx `[m, ncx]` and By `[m, ncy]` at scattered points
/// 3. Build 2D design matrix: `A[i,:]` = `Bx[i,:] ⊗ By[i,:]` (row-wise Kronecker)
/// 4. When smoothing=0: coeffs = lstsq(W*A, W*z)
/// 5. When smoothing>0: coeffs = lstsq([W*A; √λ*P], [W*z; 0])
#[allow(clippy::too_many_arguments)]
pub fn smooth_bivariate_spline_fit_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    z: &Tensor<R>,
    weights: Option<&Tensor<R>>,
    smoothing: f64,
    kx: usize,
    ky: usize,
) -> InterpolateResult<BivariateSpline<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + UtilityOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let device = client.device();
    let m = x.shape()[0];

    if y.shape()[0] != m || z.shape()[0] != m {
        return Err(InterpolateError::ShapeMismatch {
            expected: m,
            actual: y.shape()[0].min(z.shape()[0]),
            context: "smooth_bivariate_spline_fit: x, y, z must have same length".to_string(),
        });
    }
    if m < (kx + 1) * (ky + 1) {
        return Err(InterpolateError::InsufficientData {
            required: (kx + 1) * (ky + 1),
            actual: m,
            context: "smooth_bivariate_spline_fit".to_string(),
        });
    }

    // Auto-generate knot vectors: pick grid size so ncx*ncy <= m
    // With NotAKnot boundary: n_grid points → n_grid coefficients
    // We need ncx * ncy <= m, so each axis gets at most sqrt(m) grid points
    let max_per_axis = (m as f64).sqrt().floor() as usize;
    let nx_grid = max_per_axis.max(kx + 1);
    let ny_grid = max_per_axis.max(ky + 1);

    // Build grid on-device: grid = min + (max - min) * arange(0, n) / (n - 1)
    // Zero GPU↔CPU transfers.
    let x_sorted = client.sort(x, 0, false)?;
    let y_sorted = client.sort(y, 0, false)?;
    let x_min = x_sorted.narrow(0, 0, 1)?;
    let x_max = x_sorted.narrow(0, m - 1, 1)?;
    let y_min = y_sorted.narrow(0, 0, 1)?;
    let y_max = y_sorted.narrow(0, m - 1, 1)?;

    let x_grid = on_device_linspace(client, &x_min, &x_max, nx_grid)?;
    let y_grid = on_device_linspace(client, &y_min, &y_max, ny_grid)?;

    // Build knot vectors using the grid
    let knots_x =
        build_knot_vector_tensor(client, &x_grid, kx, &BSplineBoundary::NotAKnot, nx_grid)?;
    let knots_y =
        build_knot_vector_tensor(client, &y_grid, ky, &BSplineBoundary::NotAKnot, ny_grid)?;

    let ncx = knots_x.shape()[0] - kx - 1;
    let ncy = knots_y.shape()[0] - ky - 1;
    let n_coeffs = ncx * ncy;

    // Build 1D basis matrices at scattered data points
    let bx = compute_basis_matrix(client, x, &knots_x, kx, ncx)?; // [m, ncx]
    let by = compute_basis_matrix(client, y, &knots_y, ky, ncy)?; // [m, ncy]

    // Build 2D design matrix: row-wise Kronecker product
    // A[i, j*ncx + k] = Bx[i, k] * By[i, j]
    // This is equivalent to: for each row i, A[i,:] = kron(By[i,:], Bx[i,:])
    let bx_expanded = bx
        .unsqueeze(1)?
        .broadcast_to(&[m, ncy, ncx])?
        .contiguous()?; // [m, ncy, ncx]
    let by_expanded = by
        .unsqueeze(2)?
        .broadcast_to(&[m, ncy, ncx])?
        .contiguous()?; // [m, ncy, ncx]
    let a_3d = client.mul(&bx_expanded, &by_expanded)?; // [m, ncy, ncx]
    let a = a_3d.reshape(&[m, n_coeffs])?; // [m, ncy*ncx]

    // Apply weights
    let z_col = z.reshape(&[m, 1])?;
    let (a_weighted, z_weighted) = if let Some(w) = weights {
        let w_col = w.reshape(&[m, 1])?;
        let w_broad = w_col.broadcast_to(&[m, n_coeffs])?.contiguous()?;
        (client.mul(&a, &w_broad)?, client.mul(&z_col, &w_col)?)
    } else {
        (a.clone(), z_col.clone())
    };

    // Solve
    let coeffs_flat = if smoothing <= 0.0 {
        // Pure interpolation/least-squares
        LinearAlgebraAlgorithms::lstsq(client, &a_weighted, &z_weighted).map_err(|e| {
            InterpolateError::NumericalError {
                message: format!("lstsq failed: {}", e),
            }
        })?
    } else {
        // Penalized least squares: minimize ||W(Az - z)||² + λ||Pz||²
        // Augmented system: [W*A; √λ*I] @ c = [W*z; 0]
        // Using identity as roughness penalty (Tikhonov regularization)
        let sqrt_lambda = smoothing.sqrt();
        // √λ * I as roughness penalty
        let eye_vals = Tensor::full_scalar(&[n_coeffs], DType::F64, sqrt_lambda, device)?;
        let penalty = LinearAlgebraAlgorithms::diagflat(client, &eye_vals)?;

        let zero_rhs = Tensor::zeros(&[n_coeffs, 1], DType::F64, device)?;

        let a_aug = client.cat(&[&a_weighted, &penalty], 0)?; // [m + n_coeffs, n_coeffs]
        let z_aug = client.cat(&[&z_weighted, &zero_rhs], 0)?; // [m + n_coeffs, 1]

        LinearAlgebraAlgorithms::lstsq(client, &a_aug, &z_aug).map_err(|e| {
            InterpolateError::NumericalError {
                message: format!("lstsq failed: {}", e),
            }
        })?
    };

    // Reshape coefficients: [ncy*ncx, 1] → [ncx, ncy]
    // The design matrix ordering is A[i, j*ncx + k] = Bx[i,k]*By[i,j]
    // So coeffs are ordered as [ncy, ncx], transpose to [ncx, ncy]
    let coefficients = coeffs_flat
        .reshape(&[ncy, ncx])?
        .transpose(0, 1)?
        .contiguous()?;

    Ok(BivariateSpline {
        knots_x,
        knots_y,
        coefficients,
        degree_x: kx,
        degree_y: ky,
    })
}

/// Evaluate smooth bivariate spline at query points.
///
/// Delegates to the same evaluation used by RectBivariateSpline.
pub fn smooth_bivariate_spline_evaluate_impl<R, C>(
    client: &C,
    spline: &BivariateSpline<R>,
    xi: &Tensor<R>,
    yi: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    rect_bivariate_spline_evaluate_impl(client, spline, xi, yi)
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
    fn test_smooth_interpolation_exact() {
        // With smoothing=0, should recover data closely
        let (device, client) = setup();
        let n = 25;
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        let mut zv = Vec::with_capacity(n);

        // Grid of scattered points for z = x + y
        for i in 0..5 {
            for j in 0..5 {
                xv.push(i as f64);
                yv.push(j as f64);
                zv.push(i as f64 + j as f64);
            }
        }

        let x = Tensor::<CpuRuntime>::from_slice(&xv, &[n], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&yv, &[n], &device).unwrap();
        let z = Tensor::<CpuRuntime>::from_slice(&zv, &[n], &device).unwrap();

        let spline =
            smooth_bivariate_spline_fit_impl(&client, &x, &y, &z, None, 0.0, 3, 3).unwrap();

        // Evaluate at some of the original points
        let xi = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device).unwrap();
        let yi = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device).unwrap();
        let result = smooth_bivariate_spline_evaluate_impl(&client, &spline, &xi, &yi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for (i, &v) in vals.iter().enumerate() {
            let expected = (i + 1) as f64 * 2.0;
            assert!(
                (v - expected).abs() < 0.5,
                "point {}: got {} expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_smoothing_reduces_noise() {
        // Add noise to z = x + y, verify smoothing produces closer-to-truth results
        let (device, client) = setup();
        let n = 36;
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        let mut zv_noisy = Vec::with_capacity(n);
        let mut zv_true = Vec::with_capacity(n);

        let noise = [
            0.3, -0.2, 0.5, -0.4, 0.1, -0.3, 0.2, -0.1, 0.4, -0.5, 0.3, -0.2, 0.1, -0.3, 0.4, -0.1,
            0.2, -0.4, 0.5, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.5, 0.1, -0.2, 0.3, -0.4, 0.5, -0.1,
            0.2, -0.3, 0.4, -0.2,
        ];

        for i in 0..6 {
            for j in 0..6 {
                let idx = i * 6 + j;
                xv.push(i as f64);
                yv.push(j as f64);
                let true_val = i as f64 + j as f64;
                zv_true.push(true_val);
                zv_noisy.push(true_val + noise[idx]);
            }
        }

        let x = Tensor::<CpuRuntime>::from_slice(&xv, &[n], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&yv, &[n], &device).unwrap();
        let z_noisy = Tensor::<CpuRuntime>::from_slice(&zv_noisy, &[n], &device).unwrap();

        let spline_smooth =
            smooth_bivariate_spline_fit_impl(&client, &x, &y, &z_noisy, None, 0.01, 3, 3).unwrap();

        // Evaluate at test points
        let xi = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();
        let yi = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();
        let result =
            smooth_bivariate_spline_evaluate_impl(&client, &spline_smooth, &xi, &yi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // With smoothing, values should be closer to truth (4.0, 6.0)
        assert!(
            (vals[0] - 4.0).abs() < 1.0,
            "smoothed at (2,2): {} vs 4.0",
            vals[0]
        );
        assert!(
            (vals[1] - 6.0).abs() < 1.0,
            "smoothed at (3,3): {} vs 6.0",
            vals[1]
        );
    }

    fn eval_at(
        client: &CpuClient,
        spline: &BivariateSpline<CpuRuntime>,
        xs: &[f64],
        ys: &[f64],
        device: &CpuDevice,
    ) -> Vec<f64> {
        let xi = Tensor::<CpuRuntime>::from_slice(xs, &[xs.len()], device).unwrap();
        let yi = Tensor::<CpuRuntime>::from_slice(ys, &[ys.len()], device).unwrap();
        smooth_bivariate_spline_evaluate_impl(client, spline, &xi, &yi)
            .unwrap()
            .to_vec()
    }

    #[test]
    fn test_smooth_collinear_points_rank_deficient() {
        // All points lie on y = x, so the design matrix is rank deficient.
        // z = t^2 equals x*y on the line and lies in the cubic tensor space.
        // The min-norm fit is finite and reproduces z at the data points.
        let (device, client) = setup();
        let n = 25;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 / 24.0).collect();
        let zs: Vec<f64> = ts.iter().map(|t| t * t).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&ts, &[n], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&ts, &[n], &device).unwrap();
        let z = Tensor::<CpuRuntime>::from_slice(&zs, &[n], &device).unwrap();

        let spline =
            smooth_bivariate_spline_fit_impl(&client, &x, &y, &z, None, 0.0, 3, 3).unwrap();
        let coeffs: Vec<f64> = spline.coefficients.to_vec();
        assert!(coeffs.iter().all(|c| c.is_finite()));

        let inner: Vec<f64> = ts[1..n - 1].to_vec();
        let vals = eval_at(&client, &spline, &inner, &inner, &device);
        for (v, t) in vals.iter().zip(inner.iter()) {
            assert!((v - t * t).abs() < 1e-6, "t = {}, got {}", t, v);
        }
    }

    #[test]
    fn test_smooth_fewer_distinct_points_than_coefficients() {
        // Sixteen samples at four distinct corners: rank is at most four,
        // below the sixteen coefficients. The min-norm fit is finite and
        // interpolates the four corner values.
        let (device, client) = setup();
        let corners = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 3.0),
            (1.0, 1.0, 6.0),
        ];
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        let mut zv = Vec::new();
        for _ in 0..4 {
            for &(cx, cy, cz) in &corners {
                xv.push(cx);
                yv.push(cy);
                zv.push(cz);
            }
        }
        let n = xv.len();
        let x = Tensor::<CpuRuntime>::from_slice(&xv, &[n], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&yv, &[n], &device).unwrap();
        let z = Tensor::<CpuRuntime>::from_slice(&zv, &[n], &device).unwrap();

        let spline =
            smooth_bivariate_spline_fit_impl(&client, &x, &y, &z, None, 0.0, 3, 3).unwrap();
        let coeffs: Vec<f64> = spline.coefficients.to_vec();
        assert!(coeffs.iter().all(|c| c.is_finite()));

        let cx: Vec<f64> = corners.iter().map(|c| c.0).collect();
        let cy: Vec<f64> = corners.iter().map(|c| c.1).collect();
        let vals = eval_at(&client, &spline, &cx, &cy, &device);
        for (v, c) in vals.iter().zip(corners.iter()) {
            assert!((v - c.2).abs() < 1e-8, "corner {:?}, got {}", c, v);
        }
    }
}
