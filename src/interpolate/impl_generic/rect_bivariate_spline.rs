//! Rect bivariate spline generic implementation (fully on-device).
//!
//! Tensor-product B-spline on rectangular grid:
//! S(x,y) = Σᵢ Σⱼ cᵢⱼ Bᵢ(x) Bⱼ(y)
//!
//! Reuses 1D B-spline basis computation from bspline.rs.
//! Zero GPU↔CPU transfers in algorithm code.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bspline::{build_knot_vector_tensor, compute_basis_matrix};
use crate::interpolate::traits::bspline::BSplineBoundary;
use crate::interpolate::traits::rect_bivariate_spline::BivariateSpline;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fit a tensor-product B-spline to rectangular grid data.
///
/// 1. Build 1D basis matrices Bx `[nx, ncx]` and By `[ny, ncy]`
/// 2. Form 2D collocation: A = kron(By, Bx) → `[nx*ny, ncx*ncy]`
/// 3. Flatten z and solve: coeffs = solve(A, z_flat)
/// 4. Reshape coefficients to `[ncx, ncy]`
#[allow(clippy::too_many_arguments)]
pub fn rect_bivariate_spline_fit_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    z: &Tensor<R>,
    degree_x: usize,
    degree_y: usize,
    boundary: &BSplineBoundary,
) -> InterpolateResult<BivariateSpline<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let nx = x.shape()[0];
    let ny = y.shape()[0];

    // Validate inputs
    if z.shape().len() != 2 || z.shape()[0] != nx || z.shape()[1] != ny {
        return Err(InterpolateError::ShapeMismatch {
            expected: nx * ny,
            actual: z.shape().iter().product(),
            context: "rect_bivariate_spline_fit: z must be [nx, ny]".to_string(),
        });
    }
    if nx <= degree_x || ny <= degree_y {
        return Err(InterpolateError::InsufficientData {
            required: (degree_x + 1).max(degree_y + 1),
            actual: nx.min(ny),
            context: "rect_bivariate_spline_fit: need at least degree+1 points per axis"
                .to_string(),
        });
    }

    // Build knot vectors
    let knots_x = build_knot_vector_tensor(client, x, degree_x, boundary, nx)?;
    let knots_y = build_knot_vector_tensor(client, y, degree_y, boundary, ny)?;

    let ncx = knots_x.shape()[0] - degree_x - 1;
    let ncy = knots_y.shape()[0] - degree_y - 1;

    // Build 1D basis matrices
    let bx = compute_basis_matrix(client, x, &knots_x, degree_x, ncx)?; // `[nx, ncx]`
    let by = compute_basis_matrix(client, y, &knots_y, degree_y, ncy)?; // `[ny, ncy]`

    // 2D collocation via Kronecker product: A = kron(By, Bx)
    let a = LinearAlgebraAlgorithms::kron(client, &by, &bx)?; // `[nx*ny, ncx*ncy]`

    // Flatten z in row-major order: `z[i,j]` → `z_flat[j*nx + i]`
    // kron(By, Bx) expects the vec(Z) to be ordered matching kron order
    // kron(By, Bx) @ vec(C) = vec(Z) where vec() is column-major (Fortran order)
    // For row-major: we need `z_flat[j*nx + i]` = `z[i, j]`
    // Transpose z to `[ny, nx]`, then flatten
    let z_t = z.transpose(0, 1)?.contiguous(); // `[ny, nx]`
    let z_flat = z_t.reshape(&[nx * ny, 1])?;

    // Solve the system
    let coeffs_flat = LinearAlgebraAlgorithms::solve(client, &a, &z_flat).map_err(|e| {
        InterpolateError::NumericalError {
            message: format!("Failed to solve bivariate spline system: {}", e),
        }
    })?;

    let coefficients = coeffs_flat
        .reshape(&[ncy, ncx])?
        .transpose(0, 1)?
        .contiguous(); // [ncx, ncy]

    Ok(BivariateSpline {
        knots_x,
        knots_y,
        coefficients,
        degree_x,
        degree_y,
    })
}

/// Evaluate bivariate spline at scattered query points.
///
/// For each query (xi`[i]`, yi`[i]`):
///   result`[i]` = Bx_row`[i,:]` @ C @ By_row`[i,:]`ᵀ
///
/// Vectorized: element-wise multiply basis rows, matmul with C, row-sum.
pub fn rect_bivariate_spline_evaluate_impl<R, C>(
    client: &C,
    spline: &BivariateSpline<R>,
    xi: &Tensor<R>,
    yi: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let m = xi.shape()[0];
    if yi.shape()[0] != m {
        return Err(InterpolateError::ShapeMismatch {
            expected: m,
            actual: yi.shape()[0],
            context: "rect_bivariate_spline_evaluate: xi and yi must have same length".to_string(),
        });
    }

    let ncx = spline.knots_x.shape()[0] - spline.degree_x - 1;
    let ncy = spline.knots_y.shape()[0] - spline.degree_y - 1;

    // Compute basis matrices at query points
    let bx = compute_basis_matrix(client, xi, &spline.knots_x, spline.degree_x, ncx)?; // `[m, ncx]`
    let by = compute_basis_matrix(client, yi, &spline.knots_y, spline.degree_y, ncy)?; // `[m, ncy]`

    // For each query point i: result`[i]` = Bx`[i,:]` @ C @ By`[i,:]`ᵀ
    // Vectorized: tmp = Bx @ C → `[m, ncy]`, then element-wise multiply with By and sum
    let tmp = client.matmul(&bx, &spline.coefficients)?; // [m, ncy]
    let product = client.mul(&tmp, &by)?; // [m, ncy]
    let result = client.sum(&product, &[1], false)?; // [m]

    Ok(result)
}

/// Evaluate bivariate spline on a grid of query points.
///
/// Returns z`[i,j]` = S(xi`[i]`, yi`[j]`) as a `[mx, my]` tensor.
pub fn rect_bivariate_spline_evaluate_grid_impl<R, C>(
    client: &C,
    spline: &BivariateSpline<R>,
    xi: &Tensor<R>,
    yi: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let ncx = spline.knots_x.shape()[0] - spline.degree_x - 1;
    let ncy = spline.knots_y.shape()[0] - spline.degree_y - 1;

    // Basis matrices
    let bx = compute_basis_matrix(client, xi, &spline.knots_x, spline.degree_x, ncx)?; // [mx, ncx]
    let by = compute_basis_matrix(client, yi, &spline.knots_y, spline.degree_y, ncy)?; // [my, ncy]

    // Z = Bx @ C @ By^T → [mx, my]
    let tmp = client.matmul(&bx, &spline.coefficients)?; // [mx, ncy]
    let by_t = by.transpose(0, 1)?.contiguous(); // [ncy, my]
    let result = client.matmul(&tmp, &by_t)?; // [mx, my]

    Ok(result)
}

/// Evaluate partial derivative of bivariate spline at query points.
///
/// ∂^(dx+dy) S / ∂x^dx ∂y^dy
///
/// Differentiates the 1D B-spline representations along each axis,
/// then evaluates the resulting lower-degree spline.
pub fn rect_bivariate_spline_partial_derivative_impl<R, C>(
    client: &C,
    spline: &BivariateSpline<R>,
    xi: &Tensor<R>,
    yi: &Tensor<R>,
    dx: usize,
    dy: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let m = xi.shape()[0];
    let device = client.device();

    if yi.shape()[0] != m {
        return Err(InterpolateError::ShapeMismatch {
            expected: m,
            actual: yi.shape()[0],
            context: "rect_bivariate_spline_partial_derivative".to_string(),
        });
    }

    // If derivative order exceeds degree, result is zero
    if dx > spline.degree_x || dy > spline.degree_y {
        return Ok(Tensor::zeros(&[m], DType::F64, device));
    }

    // Differentiate along x: apply knot differencing dx times to rows of C
    // This transforms C [ncx, ncy] → C_dx [ncx-dx, ncy] with lower degree knots
    let (knots_x_d, coeffs_dx, degree_x_d) = differentiate_2d_x(
        client,
        &spline.knots_x,
        &spline.coefficients,
        spline.degree_x,
        dx,
    )?;

    // Differentiate along y: apply knot differencing dy times to columns of C_dx
    let (knots_y_d, coeffs_dxy, degree_y_d) =
        differentiate_2d_y(client, &spline.knots_y, &coeffs_dx, spline.degree_y, dy)?;

    let ncx_d = knots_x_d.shape()[0] - degree_x_d - 1;
    let ncy_d = knots_y_d.shape()[0] - degree_y_d - 1;

    // Evaluate with differentiated spline
    let bx = compute_basis_matrix(client, xi, &knots_x_d, degree_x_d, ncx_d)?;
    let by = compute_basis_matrix(client, yi, &knots_y_d, degree_y_d, ncy_d)?;

    let tmp = client.matmul(&bx, &coeffs_dxy)?;
    let product = client.mul(&tmp, &by)?;
    let result = client.sum(&product, &[1], false)?;

    Ok(result)
}

/// Integrate bivariate spline over [xa, xb] × [ya, yb].
///
/// Uses separability: ∫∫ S(x,y) dx dy = (∫ Bx dx) @ C @ (∫ By dy)ᵀ
pub fn rect_bivariate_spline_integrate_impl<R, C>(
    client: &C,
    spline: &BivariateSpline<R>,
    xa: f64,
    xb: f64,
    ya: f64,
    yb: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let ncx = spline.knots_x.shape()[0] - spline.degree_x - 1;
    let ncy = spline.knots_y.shape()[0] - spline.degree_y - 1;

    // Compute definite integrals of each 1D basis function over [xa,xb] and [ya,yb]
    let ix = integrate_basis(client, &spline.knots_x, spline.degree_x, ncx, xa, xb)?; // [ncx]
    let iy = integrate_basis(client, &spline.knots_y, spline.degree_y, ncy, ya, yb)?; // [ncy]

    // Result = ix^T @ C @ iy (scalar)
    let ix_row = ix.reshape(&[1, ncx])?;
    let iy_col = iy.reshape(&[ncy, 1])?;
    let tmp = client.matmul(&ix_row, &spline.coefficients)?; // [1, ncy]
    let result = client.matmul(&tmp, &iy_col)?; // [1, 1]
    Ok(result.reshape(&[1])?)
}

// ============ Helpers ============

/// Differentiate 2D coefficients along x-axis (rows).
///
/// Applies 1D B-spline differentiation to each column of C independently.
/// C [ncx, ncy] → C' [ncx-order, ncy]
fn differentiate_2d_x<R, C>(
    client: &C,
    knots_x: &Tensor<R>,
    coefficients: &Tensor<R>,
    degree_x: usize,
    order: usize,
) -> InterpolateResult<(Tensor<R>, Tensor<R>, usize)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if order == 0 {
        return Ok((knots_x.clone(), coefficients.clone(), degree_x));
    }

    let ncy = coefficients.shape()[1];

    let mut current_knots = knots_x.clone();
    let mut current_coeffs = coefficients.clone(); // [ncx_cur, ncy]
    let mut current_degree = degree_x;

    for _ in 0..order {
        if current_degree == 0 {
            break;
        }
        let n = current_coeffs.shape()[0];
        let k = current_degree;
        let n_knots = current_knots.shape()[0];

        // c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
        let c_hi = current_coeffs.narrow(0, 1, n - 1)?.contiguous(); // [n-1, ncy]
        let c_lo = current_coeffs.narrow(0, 0, n - 1)?.contiguous(); // [n-1, ncy]
        let dc = client.sub(&c_hi, &c_lo)?; // [n-1, ncy]

        let t_hi = current_knots.narrow(0, k + 1, n - 1)?.contiguous(); // [n-1]
        let t_lo = current_knots.narrow(0, 1, n - 1)?.contiguous(); // [n-1]
        let dt = client.sub(&t_hi, &t_lo)?; // [n-1]

        // Safe division: broadcast dt to [n-1, ncy]
        let dt_col = dt
            .reshape(&[n - 1, 1])?
            .broadcast_to(&[n - 1, ncy])?
            .contiguous();
        let eps = Tensor::full_scalar(&[n - 1, ncy], DType::F64, 1e-300, client.device());
        let abs_dt = client.abs(&dt_col)?;
        let dt_safe = client.maximum(&abs_dt, &eps)?;
        let zero = Tensor::zeros(&[n - 1, ncy], DType::F64, client.device());
        let mask = client.gt(&abs_dt, &zero)?;

        let new_coeffs =
            client.mul_scalar(&client.mul(&client.div(&dc, &dt_safe)?, &mask)?, k as f64)?;

        let new_knots = current_knots.narrow(0, 1, n_knots - 2)?.contiguous();

        current_coeffs = new_coeffs;
        current_knots = new_knots;
        current_degree -= 1;
    }

    Ok((current_knots, current_coeffs, current_degree))
}

/// Differentiate 2D coefficients along y-axis (columns).
///
/// Transposes, differentiates along rows (now columns), transposes back.
fn differentiate_2d_y<R, C>(
    client: &C,
    knots_y: &Tensor<R>,
    coefficients: &Tensor<R>,
    degree_y: usize,
    order: usize,
) -> InterpolateResult<(Tensor<R>, Tensor<R>, usize)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if order == 0 {
        return Ok((knots_y.clone(), coefficients.clone(), degree_y));
    }

    // Transpose: [ncx, ncy] → [ncy, ncx], differentiate along "x" (which is really y), transpose back
    let c_t = coefficients.transpose(0, 1)?.contiguous();
    let (knots_d, c_d, degree_d) = differentiate_2d_x(client, knots_y, &c_t, degree_y, order)?;
    let c_result = c_d.transpose(0, 1)?.contiguous();

    Ok((knots_d, c_result, degree_d))
}

/// Compute definite integrals of each 1D B-spline basis function over [a, b].
///
/// Returns tensor [n_coeffs] where entry i is ∫_a^b B_i(x) dx.
///
/// Vectorized: builds anti-derivative coefficients for ALL basis functions
/// at once using diagflat + cumsum, then evaluates in a single matmul.
/// Zero GPU↔CPU transfers.
fn integrate_basis<R, C>(
    client: &C,
    knots: &Tensor<R>,
    degree: usize,
    n_coeffs: usize,
    a: f64,
    b: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let device = client.device();
    let k = degree;
    let n_knots = knots.shape()[0];

    // Anti-derivative B-spline knots: prepend first, append last
    let first = knots.narrow(0, 0, 1)?.contiguous();
    let last = knots.narrow(0, n_knots - 1, 1)?.contiguous();
    let anti_knots = client.cat(&[&first, knots, &last], 0)?;
    let ncx_anti = anti_knots.shape()[0] - (k + 1) - 1;

    // Knot differences: dt[i] = t[i+k+1] - t[i], scaled by 1/(k+1)
    let t_hi = knots.narrow(0, k + 1, n_coeffs)?.contiguous();
    let t_lo = knots.narrow(0, 0, n_coeffs)?.contiguous();
    let dt_scaled = client.mul_scalar(&client.sub(&t_hi, &t_lo)?, 1.0 / (k + 1) as f64)?;

    // For basis function i with coefficient vector e_i (identity column):
    //   terms_i = e_i * dt_scaled = dt_scaled[i] at position i, 0 elsewhere
    //   anti_coeffs_i = [0, cumsum(terms_i)]
    //
    // For ALL basis functions at once:
    //   terms_matrix = diagflat(dt_scaled)  → [n_coeffs, n_coeffs]
    //   cumsum along dim=0 → [n_coeffs, n_coeffs]
    //   prepend zero row → [n_coeffs+1, n_coeffs]
    let terms_matrix = LinearAlgebraAlgorithms::diagflat(client, &dt_scaled)?; // [n_coeffs, n_coeffs]
    let cumsum_matrix = client.cumsum(&terms_matrix, 0)?; // [n_coeffs, n_coeffs]
    let zero_row = Tensor::zeros(&[1, n_coeffs], DType::F64, device);
    let anti_coeffs_all = client.cat(&[&zero_row, &cumsum_matrix], 0)?; // [n_coeffs+1, n_coeffs]

    // Evaluate anti-derivative basis at [b, a]: shared basis matrix [2, ncx_anti]
    let ab = Tensor::from_slice(&[b, a], &[2], device);
    let basis_ab = compute_basis_matrix(client, &ab, &anti_knots, k + 1, ncx_anti)?; // [2, ncx_anti]

    // Matmul: [2, ncx_anti] @ [ncx_anti, n_coeffs] → [2, n_coeffs]
    let vals = client.matmul(&basis_ab, &anti_coeffs_all)?; // [2, n_coeffs]

    // Integral = F(b) - F(a) for each basis function
    let vals_b = vals.narrow(0, 0, 1)?.contiguous().reshape(&[n_coeffs])?;
    let vals_a = vals.narrow(0, 1, 1)?.contiguous().reshape(&[n_coeffs])?;
    Ok(client.sub(&vals_b, &vals_a)?)
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
    fn test_bilinear_exact() {
        // z = x + 2y should be exact with degree 1
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let mut z_data = vec![0.0f64; 12];
        for i in 0..4 {
            for j in 0..3 {
                z_data[i * 3 + j] = i as f64 + 2.0 * j as f64;
            }
        }
        let z = Tensor::<CpuRuntime>::from_slice(&z_data, &[4, 3], &device);

        let spline =
            rect_bivariate_spline_fit_impl(&client, &x, &y, &z, 1, 1, &BSplineBoundary::NotAKnot)
                .expect("fit failed");

        // Evaluate at scattered query points
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let yi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.0, 1.5], &[3], &device);
        let result = rect_bivariate_spline_evaluate_impl(&client, &spline, &xi, &yi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        let expected = [0.5 + 1.0, 1.5 + 2.0, 2.5 + 3.0];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (v - e).abs() < 1e-8,
                "point {}: got {} expected {}",
                i,
                v,
                e
            );
        }
    }

    #[test]
    fn test_grid_evaluation() {
        // z = x * y
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);

        let mut z_data = vec![0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                z_data[i * 4 + j] = i as f64 * j as f64;
            }
        }
        let z = Tensor::<CpuRuntime>::from_slice(&z_data, &[4, 4], &device);

        let spline =
            rect_bivariate_spline_fit_impl(&client, &x, &y, &z, 3, 3, &BSplineBoundary::NotAKnot)
                .expect("fit failed");

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5], &[2], &device);
        let yi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let grid = rect_bivariate_spline_evaluate_grid_impl(&client, &spline, &xi, &yi).unwrap();

        assert_eq!(grid.shape(), &[2, 3]);
        let vals: Vec<f64> = grid.to_vec();

        // z = x*y: grid[0,0] = 0.5*0.5 = 0.25, grid[1,2] = 1.5*2.5 = 3.75
        assert!(
            (vals[0] - 0.25).abs() < 0.1,
            "grid[0,0]: {} vs 0.25",
            vals[0]
        );
        assert!(
            (vals[5] - 3.75).abs() < 0.1,
            "grid[1,2]: {} vs 3.75",
            vals[5]
        );
    }

    #[test]
    fn test_partial_derivative() {
        // z = x^2 + y^2, dz/dx = 2x, dz/dy = 2y
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);

        let mut z_data = vec![0.0f64; 25];
        for i in 0..5 {
            for j in 0..5 {
                z_data[i * 5 + j] = (i * i + j * j) as f64;
            }
        }
        let z = Tensor::<CpuRuntime>::from_slice(&z_data, &[5, 5], &device);

        let spline =
            rect_bivariate_spline_fit_impl(&client, &x, &y, &z, 3, 3, &BSplineBoundary::NotAKnot)
                .expect("fit failed");

        // dz/dx at (2.0, 1.0) should be ~4.0
        let xi = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device);
        let yi = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let dzdx = rect_bivariate_spline_partial_derivative_impl(&client, &spline, &xi, &yi, 1, 0)
            .unwrap();
        let val: Vec<f64> = dzdx.to_vec();
        assert!(
            (val[0] - 4.0).abs() < 0.5,
            "dz/dx at (2,1): {} vs 4.0",
            val[0]
        );

        // dz/dy at (1.0, 2.0) should be ~4.0
        let xi2 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yi2 = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device);
        let dzdy =
            rect_bivariate_spline_partial_derivative_impl(&client, &spline, &xi2, &yi2, 0, 1)
                .unwrap();
        let val2: Vec<f64> = dzdy.to_vec();
        assert!(
            (val2[0] - 4.0).abs() < 0.5,
            "dz/dy at (1,2): {} vs 4.0",
            val2[0]
        );
    }

    #[test]
    fn test_integrate_constant() {
        // z = 1.0 everywhere, integral over [0,3]×[0,2] should be 6.0
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 12], &[4, 3], &device);

        let spline =
            rect_bivariate_spline_fit_impl(&client, &x, &y, &z, 1, 1, &BSplineBoundary::NotAKnot)
                .expect("fit failed");

        let result =
            rect_bivariate_spline_integrate_impl(&client, &spline, 0.0, 3.0, 0.0, 2.0).unwrap();
        let val: Vec<f64> = result.to_vec();
        assert!((val[0] - 6.0).abs() < 1e-6, "integral: {} vs 6.0", val[0]);
    }
}
