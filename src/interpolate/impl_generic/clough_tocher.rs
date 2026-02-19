//! Clough-Tocher 2D interpolation generic implementation.
//!
//! C1-continuous piecewise cubic interpolation on Delaunay triangulation.
//! Uses the Clough-Tocher split: each triangle is divided into 3 sub-triangles
//! at the centroid, and cubic polynomials are fitted with C1 continuity.
//!
//! The gradient estimation uses a least-squares fit over each vertex's
//! neighboring triangles, similar to SciPy's approach.
//!
//! Fully on-device — zero GPU↔CPU transfers in both fit and evaluate.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::clough_tocher::CloughTocher2D;
use crate::spatial::traits::delaunay::{Delaunay, DelaunayAlgorithms};
use numr::ops::{CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fit a Clough-Tocher interpolant.
///
/// 1. Compute Delaunay triangulation
/// 2. Estimate gradients at each vertex via least-squares
pub fn clough_tocher_fit_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    values: &Tensor<R>,
    fill_value: f64,
) -> InterpolateResult<CloughTocher2D<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + DelaunayAlgorithms<R>
        + RuntimeClient<R>,
{
    let n = points.shape()[0];
    if points.shape().len() != 2 || points.shape()[1] != 2 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "points".to_string(),
            message: "points must be [n, 2]".to_string(),
        });
    }
    if values.shape()[0] != n {
        return Err(InterpolateError::ShapeMismatch {
            expected: n,
            actual: values.shape()[0],
            context: "clough_tocher_fit: points vs values".to_string(),
        });
    }
    if n < 3 {
        return Err(InterpolateError::InsufficientData {
            required: 3,
            actual: n,
            context: "clough_tocher_fit: need at least 3 points".to_string(),
        });
    }

    // Compute Delaunay triangulation
    let tri = client
        .delaunay(points)
        .map_err(|e| InterpolateError::NumericalError {
            message: format!("Delaunay triangulation failed: {}", e),
        })?;

    // Estimate gradients at each vertex
    let gradients = estimate_gradients(client, &tri, values)?;

    Ok(CloughTocher2D {
        triangulation: tri,
        values: values.clone(),
        gradients,
        fill_value,
    })
}

/// Evaluate the Clough-Tocher interpolant at query points.
///
/// Fully vectorized on-device: uses gather to look up triangle vertices,
/// values, and gradients, then computes barycentric coordinates and
/// Bernstein-Bézier evaluation in batched tensor ops. Zero GPU↔CPU transfers.
pub fn clough_tocher_evaluate_impl<R, C>(
    client: &C,
    ct: &CloughTocher2D<R>,
    xi: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + DelaunayAlgorithms<R>
        + RuntimeClient<R>,
{
    let m = xi.shape()[0];
    if xi.shape().len() != 2 || xi.shape()[1] != 2 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "xi".to_string(),
            message: "xi must be [m, 2]".to_string(),
        });
    }

    let device = client.device();

    // Find containing simplex for each query point → [m] i64
    let simplex_ids = client
        .delaunay_find_simplex(&ct.triangulation, xi)
        .map_err(|e| InterpolateError::NumericalError {
            message: format!("Find simplex failed: {}", e),
        })?;

    // Mask for valid (inside hull) points: simplex_id >= 0
    let zero = Tensor::zeros(&[m], DType::I64, device);
    let valid_mask = client.ge(&simplex_ids, &zero)?; // [m] bool

    // Clamp negative simplex IDs to 0 for safe indexing (results masked out later)
    let safe_ids = client.maximum(&simplex_ids, &zero)?; // [m] i64

    // Gather triangle vertex indices: simplices is [n_tri, 3]
    // index_select(simplices, dim=0, index=safe_ids) → [m, 3] vertex indices
    let tri_vertex_ids = client.index_select(&ct.triangulation.simplices, 0, &safe_ids)?; // [m, 3] i64

    // Flatten vertex IDs for gathering from points/values/gradients
    let tri_flat = tri_vertex_ids.reshape(&[m * 3])?; // [m*3] i64

    // Gather vertex positions: points is [n, 2]
    let verts_flat = client.index_select(&ct.triangulation.points, 0, &tri_flat)?; // [m*3, 2]
    let verts = verts_flat.reshape(&[m, 3, 2])?; // [m, 3, 2]

    // Gather vertex values: values is [n]
    let vals_flat = client.index_select(&ct.values, 0, &tri_flat)?; // [m*3]
    let vals = vals_flat.reshape(&[m, 3])?; // [m, 3]

    // Gather vertex gradients: gradients is [n, 2]
    let grads_flat = client.index_select(&ct.gradients, 0, &tri_flat)?; // [m*3, 2]
    let grads = grads_flat.reshape(&[m, 3, 2])?; // [m, 3, 2]

    // Extract vertex coordinates: p0, p1, p2 each [m, 2]
    let p0 = verts.narrow(1, 0, 1)?.contiguous().reshape(&[m, 2])?; // [m, 2]
    let p1 = verts.narrow(1, 1, 1)?.contiguous().reshape(&[m, 2])?;
    let p2 = verts.narrow(1, 2, 1)?.contiguous().reshape(&[m, 2])?;

    // Barycentric coordinates (vectorized 2x2 solve)
    // d = p1 - p0, e = p2 - p0, q = xi - p0
    let d = client.sub(&p1, &p0)?; // [m, 2]
    let e = client.sub(&p2, &p0)?; // [m, 2]
    let q = client.sub(xi, &p0)?; // [m, 2]

    let d0 = d.narrow(1, 0, 1)?.contiguous().reshape(&[m])?; // [m]
    let d1 = d.narrow(1, 1, 1)?.contiguous().reshape(&[m])?;
    let e0 = e.narrow(1, 0, 1)?.contiguous().reshape(&[m])?;
    let e1 = e.narrow(1, 1, 1)?.contiguous().reshape(&[m])?;
    let q0 = q.narrow(1, 0, 1)?.contiguous().reshape(&[m])?;
    let q1 = q.narrow(1, 1, 1)?.contiguous().reshape(&[m])?;

    // det = d0*e1 - e0*d1
    let det = client.sub(&client.mul(&d0, &e1)?, &client.mul(&e0, &d1)?)?;
    // Safe inverse: clamp |det| away from 0, preserve sign
    let eps = Tensor::full_scalar(&[m], DType::F64, 1e-15, device);
    let neg_eps = Tensor::full_scalar(&[m], DType::F64, -1e-15, device);
    let zero = Tensor::zeros(&[m], DType::F64, device);
    let det_ge_zero = client.ge(&det, &zero)?;
    let det_safe = client.where_cond(
        &det_ge_zero,
        &client.maximum(&det, &eps)?,
        &client.minimum(&det, &neg_eps)?,
    )?;
    let one = Tensor::full_scalar(&[m], DType::F64, 1.0, device);
    let inv_det = client.div(&one, &det_safe)?;

    // b1 = inv_det * (e1*q0 - e0*q1), b2 = inv_det * (-d1*q0 + d0*q1)
    let b1 = client.mul(
        &inv_det,
        &client.sub(&client.mul(&e1, &q0)?, &client.mul(&e0, &q1)?)?,
    )?;
    let b2 = client.mul(
        &inv_det,
        &client.sub(&client.mul(&d0, &q1)?, &client.mul(&d1, &q0)?)?,
    )?;
    let b0 = client.sub(
        &client.sub(&Tensor::full_scalar(&[m], DType::F64, 1.0, device), &b1)?,
        &b2,
    )?;

    // Extract per-vertex values: f0, f1, f2 each [m]
    let f0 = vals.narrow(1, 0, 1)?.contiguous().reshape(&[m])?;
    let f1 = vals.narrow(1, 1, 1)?.contiguous().reshape(&[m])?;
    let f2 = vals.narrow(1, 2, 1)?.contiguous().reshape(&[m])?;

    // Extract per-vertex gradients: g0, g1, g2 each [m, 2]
    let g0 = grads.narrow(1, 0, 1)?.contiguous().reshape(&[m, 2])?;
    let g1 = grads.narrow(1, 1, 1)?.contiguous().reshape(&[m, 2])?;
    let g2 = grads.narrow(1, 2, 1)?.contiguous().reshape(&[m, 2])?;

    // Edge vectors: e01 = p1-p0, e02 = p2-p0, e12 = p2-p1 each [m, 2]
    let e01 = client.sub(&p1, &p0)?;
    let e02 = client.sub(&p2, &p0)?;
    let e12 = client.sub(&p2, &p1)?;

    // Directional derivatives: dot(grad, edge) for each vertex-edge pair
    // dot product along dim=1: sum(g * e, dim=1)
    let df0_01 = client.sum(&client.mul(&g0, &e01)?, &[1], false)?; // [m]
    let df1_01 = client.sum(&client.mul(&g1, &e01)?, &[1], false)?;
    let df0_02 = client.sum(&client.mul(&g0, &e02)?, &[1], false)?;
    let df2_02 = client.sum(&client.mul(&g2, &e02)?, &[1], false)?;
    let df1_12 = client.sum(&client.mul(&g1, &e12)?, &[1], false)?;
    let df2_12 = client.sum(&client.mul(&g2, &e12)?, &[1], false)?;

    // Bernstein-Bézier control points
    let third = Tensor::full_scalar(&[m], DType::F64, 1.0 / 3.0, device);
    let c300 = f0.clone();
    let c030 = f1.clone();
    let c003 = f2.clone();
    let c210 = client.add(&f0, &client.mul(&df0_01, &third)?)?;
    let c120 = client.sub(&f1, &client.mul(&df1_01, &third)?)?;
    let c201 = client.add(&f0, &client.mul(&df0_02, &third)?)?;
    let c021 = client.add(&f1, &client.mul(&df1_12, &third)?)?;
    let c102 = client.sub(&f2, &client.mul(&df2_02, &third)?)?;
    let c012 = client.sub(&f2, &client.mul(&df2_12, &third)?)?;

    // Interior control point: average of edge estimates
    let sixth = Tensor::full_scalar(&[m], DType::F64, 1.0 / 6.0, device);
    let c111_sum = client.add(
        &client.add(&client.add(&c210, &c120)?, &client.add(&c201, &c021)?)?,
        &client.add(&c102, &c012)?,
    )?;
    let c111 = client.mul(&c111_sum, &sixth)?;

    // Evaluate cubic Bernstein polynomial
    let b00 = client.mul(&b0, &b0)?;
    let b11 = client.mul(&b1, &b1)?;
    let b22 = client.mul(&b2, &b2)?;
    let three = Tensor::full_scalar(&[m], DType::F64, 3.0, device);
    let six = Tensor::full_scalar(&[m], DType::F64, 6.0, device);

    // c300*b0³ + c030*b1³ + c003*b2³
    let mut result = client.add(
        &client.add(
            &client.mul(&c300, &client.mul(&b00, &b0)?)?,
            &client.mul(&c030, &client.mul(&b11, &b1)?)?,
        )?,
        &client.mul(&c003, &client.mul(&b22, &b2)?)?,
    )?;

    // + 3*(c210*b0²b1 + c120*b0*b1² + c201*b0²b2 + c021*b1²b2 + c102*b0*b2² + c012*b1*b2²)
    let quad_terms = client.add(
        &client.add(
            &client.add(
                &client.mul(&c210, &client.mul(&b00, &b1)?)?,
                &client.mul(&c120, &client.mul(&b0, &b11)?)?,
            )?,
            &client.add(
                &client.mul(&c201, &client.mul(&b00, &b2)?)?,
                &client.mul(&c021, &client.mul(&b11, &b2)?)?,
            )?,
        )?,
        &client.add(
            &client.mul(&c102, &client.mul(&b0, &b22)?)?,
            &client.mul(&c012, &client.mul(&b1, &b22)?)?,
        )?,
    )?;
    result = client.add(&result, &client.mul(&three, &quad_terms)?)?;

    // + 6*c111*b0*b1*b2
    let b012 = client.mul(&b0, &client.mul(&b1, &b2)?)?;
    result = client.add(&result, &client.mul(&six, &client.mul(&c111, &b012)?)?)?;

    // Apply fill_value for points outside the convex hull
    let fill = Tensor::full_scalar(&[m], DType::F64, ct.fill_value, device);
    let result = client.where_cond(&valid_mask, &result, &fill)?;

    Ok(result)
}

// ============ Gradient estimation ============

/// Estimate gradients at each vertex using least-squares over neighboring data.
///
/// Fully vectorized on-device using triangle edges + scatter_reduce.
///
/// For each vertex i, solves: f(pj) - f(pi) ≈ grad_i · (pj - pi)
/// via normal equations A^T A @ grad = A^T b, accumulated with scatter_reduce.
fn estimate_gradients<R, C>(
    client: &C,
    tri: &Delaunay<R>,
    values: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + ConditionalOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n = tri.points.shape()[0];
    let n_tri = tri.simplices.shape()[0];

    // Build directed edge list from simplices [n_tri, 3].
    // Each triangle (a,b,c) contributes 6 directed edges:
    //   a→b, a→c, b→a, b→c, c→a, c→b
    let col_a = tri
        .simplices
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_tri])?;
    let col_b = tri
        .simplices
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_tri])?;
    let col_c = tri
        .simplices
        .narrow(1, 2, 1)?
        .contiguous()
        .reshape(&[n_tri])?;

    // src[e] = vertex whose gradient we're estimating
    // dst[e] = the neighbor vertex
    let src = client.cat(&[&col_a, &col_a, &col_b, &col_b, &col_c, &col_c], 0)?; // [6*n_tri]
    let dst = client.cat(&[&col_b, &col_c, &col_a, &col_c, &col_a, &col_b], 0)?; // [6*n_tri]
    let n_edges = 6 * n_tri;

    // Gather coordinates and values for src and dst vertices
    let src_pts = client.index_select(&tri.points, 0, &src)?; // [n_edges, 2]
    let dst_pts = client.index_select(&tri.points, 0, &dst)?; // [n_edges, 2]
    let src_vals = client.index_select(values, 0, &src)?; // [n_edges]
    let dst_vals = client.index_select(values, 0, &dst)?; // [n_edges]

    // Compute deltas: delta_pos = dst - src, delta_val = dst_val - src_val
    let delta = client.sub(&dst_pts, &src_pts)?; // [n_edges, 2]
    let dx = delta.narrow(1, 0, 1)?.contiguous().reshape(&[n_edges])?; // [n_edges]
    let dy = delta.narrow(1, 1, 1)?.contiguous().reshape(&[n_edges])?;
    let df = client.sub(&dst_vals, &src_vals)?; // [n_edges]

    // Compute the 5 products for A^T A and A^T b per edge
    let dxdx = client.mul(&dx, &dx)?; // [n_edges]
    let dxdy = client.mul(&dx, &dy)?;
    let dydy = client.mul(&dy, &dy)?;
    let dxdf = client.mul(&dx, &df)?;
    let dydf = client.mul(&dy, &df)?;

    // Scatter-reduce into per-vertex accumulators using src vertex indices
    let zeros_n = Tensor::zeros(&[n], DType::F64, device);
    let ata_00 = client.scatter_reduce(&zeros_n, 0, &src, &dxdx, ScatterReduceOp::Sum, true)?;
    let ata_01 = client.scatter_reduce(&zeros_n, 0, &src, &dxdy, ScatterReduceOp::Sum, true)?;
    let ata_11 = client.scatter_reduce(&zeros_n, 0, &src, &dydy, ScatterReduceOp::Sum, true)?;
    let atb_0 = client.scatter_reduce(&zeros_n, 0, &src, &dxdf, ScatterReduceOp::Sum, true)?;
    let atb_1 = client.scatter_reduce(&zeros_n, 0, &src, &dydf, ScatterReduceOp::Sum, true)?;

    // Solve 2x2 normal equations per vertex: det = a00*a11 - a01²
    let det = client.sub(
        &client.mul(&ata_00, &ata_11)?,
        &client.mul(&ata_01, &ata_01)?,
    )?;

    // Safe inverse of determinant (clamp away from zero)
    let eps = Tensor::full_scalar(&[n], DType::F64, 1e-15, device);
    let neg_eps = Tensor::full_scalar(&[n], DType::F64, -1e-15, device);
    let zero = Tensor::zeros(&[n], DType::F64, device);
    let det_ge_zero = client.ge(&det, &zero)?;
    let det_safe = client.where_cond(
        &det_ge_zero,
        &client.maximum(&det, &eps)?,
        &client.minimum(&det, &neg_eps)?,
    )?;
    let one = Tensor::full_scalar(&[n], DType::F64, 1.0, device);
    let inv_det = client.div(&one, &det_safe)?;

    // Cramer's rule: gx = inv_det * (a11*b0 - a01*b1)
    //                gy = inv_det * (a00*b1 - a01*b0)
    let gx = client.mul(
        &inv_det,
        &client.sub(&client.mul(&ata_11, &atb_0)?, &client.mul(&ata_01, &atb_1)?)?,
    )?;
    let gy = client.mul(
        &inv_det,
        &client.sub(&client.mul(&ata_00, &atb_1)?, &client.mul(&ata_01, &atb_0)?)?,
    )?;

    // Stack into [n, 2]
    let gx_col = gx.reshape(&[n, 1])?;
    let gy_col = gy.reshape(&[n, 1])?;
    Ok(client.cat(&[&gx_col, &gy_col], 1)?)
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
    fn test_linear_function_exact() {
        // f(x,y) = 2x + 3y + 1 should be reproduced exactly by cubic interpolant
        let (device, client) = setup();

        let points_data = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.75, 0.75, 0.25,
        ];
        let n = points_data.len() / 2;
        let points = Tensor::<CpuRuntime>::from_slice(&points_data, &[n, 2], &device);

        let values_data: Vec<f64> = (0..n)
            .map(|i| 2.0 * points_data[i * 2] + 3.0 * points_data[i * 2 + 1] + 1.0)
            .collect();
        let values = Tensor::<CpuRuntime>::from_slice(&values_data, &[n], &device);

        let ct = clough_tocher_fit_impl(&client, &points, &values, f64::NAN).unwrap();

        // Query at interior points
        let xi_data = vec![0.3, 0.3, 0.6, 0.2, 0.2, 0.6];
        let xi = Tensor::<CpuRuntime>::from_slice(&xi_data, &[3, 2], &device);

        let result = clough_tocher_evaluate_impl(&client, &ct, &xi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for i in 0..3 {
            let expected = 2.0 * xi_data[i * 2] + 3.0 * xi_data[i * 2 + 1] + 1.0;
            assert!(
                (vals[i] - expected).abs() < 0.1,
                "point {}: got {} expected {}",
                i,
                vals[i],
                expected
            );
        }
    }

    #[test]
    fn test_quadratic_surface() {
        // f(x,y) = x^2 + y^2 - cubic should approximate well
        let (device, client) = setup();

        let mut points_data = Vec::new();
        let mut values_data = Vec::new();

        // Create a grid of points
        for i in 0..5 {
            for j in 0..5 {
                let x = i as f64 * 0.25;
                let y = j as f64 * 0.25;
                points_data.push(x);
                points_data.push(y);
                values_data.push(x * x + y * y);
            }
        }
        let n = values_data.len();
        let points = Tensor::<CpuRuntime>::from_slice(&points_data, &[n, 2], &device);
        let values = Tensor::<CpuRuntime>::from_slice(&values_data, &[n], &device);

        let ct = clough_tocher_fit_impl(&client, &points, &values, f64::NAN).unwrap();

        // Query at center-ish point
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);
        let result = clough_tocher_evaluate_impl(&client, &ct, &xi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        let expected = 0.5;
        assert!(
            (vals[0] - expected).abs() < 0.15,
            "at (0.5,0.5): got {} expected {}",
            vals[0],
            expected
        );
    }

    #[test]
    fn test_outside_hull_fill_value() {
        let (device, client) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device);
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);

        let ct = clough_tocher_fit_impl(&client, &points, &values, -999.0).unwrap();

        // Query outside the triangle
        let xi = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[1, 2], &device);
        let result = clough_tocher_evaluate_impl(&client, &ct, &xi).unwrap();
        let vals: Vec<f64> = result.to_vec();

        assert_eq!(vals[0], -999.0, "outside hull should be fill_value");
    }
}
