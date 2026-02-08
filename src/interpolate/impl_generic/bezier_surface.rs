//! Bezier surface generic implementation (fully on-device).
//!
//! Uses tensor product of 1D Bernstein bases for evaluation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bezier_curve::bernstein_basis_matrix;
use crate::interpolate::traits::bezier_surface::BezierSurface;
use numr::algorithm::special::SpecialFunctions;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Evaluate a Bezier surface at (u, v) parameter pairs (fully on-device).
///
/// S(u,v) = sum_ij B_i(u) * B_j(v) * P_ij
/// = basis_u @ P_reshaped @ basis_v^T (per eval point)
///
/// For batch: build full tensor product basis and matmul.
pub fn bezier_surface_evaluate_impl<R, C>(
    client: &C,
    surface: &BezierSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    let shape = surface.control_points.shape();
    if shape.len() != 3 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "control_points".to_string(),
            message: "expected shape [nu, nv, n_dims]".to_string(),
        });
    }
    let nu = shape[0];
    let nv = shape[1];
    let n_dims = shape[2];
    let m = u.shape()[0];

    if u.shape()[0] != v.shape()[0] {
        return Err(InterpolateError::ShapeMismatch {
            expected: u.shape()[0],
            actual: v.shape()[0],
            context: "bezier_surface: u and v must have same length".to_string(),
        });
    }

    // Bernstein bases: [m, nu] and [m, nv]
    let basis_u = bernstein_basis_matrix(client, u, surface.degree_u)?;
    let basis_v = bernstein_basis_matrix(client, v, surface.degree_v)?;

    // Tensor product: for each eval point, weight = basis_u[i] * basis_v[j]
    // Flatten control points to [nu*nv, n_dims]
    let cp_flat = surface
        .control_points
        .reshape(&[nu * nv, n_dims])?
        .contiguous();

    // Build tensor product basis [m, nu*nv]
    // basis_u: [m, nu], basis_v: [m, nv]
    // product[k, i*nv+j] = basis_u[k, i] * basis_v[k, j]
    let bu_exp = basis_u
        .reshape(&[m, nu, 1])?
        .broadcast_to(&[m, nu, nv])?
        .contiguous();
    let bv_exp = basis_v
        .reshape(&[m, 1, nv])?
        .broadcast_to(&[m, nu, nv])?
        .contiguous();
    let product = client.mul(&bu_exp, &bv_exp)?;
    let product_flat = product.reshape(&[m, nu * nv])?;

    // result = product_flat @ cp_flat → [m, n_dims]
    let result = client.matmul(&product_flat, &cp_flat)?;
    Ok(result)
}

/// Evaluate partial derivatives of a Bezier surface (fully on-device).
///
/// Uses derivative of Bernstein basis in each direction.
pub fn bezier_surface_partial_impl<R, C>(
    client: &C,
    surface: &BezierSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
    du: usize,
    dv: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    if du == 0 && dv == 0 {
        return bezier_surface_evaluate_impl(client, surface, u, v);
    }

    let shape = surface.control_points.shape();
    let nu = shape[0];
    let nv = shape[1];
    let n_dims = shape[2];
    let m = u.shape()[0];
    let device = client.device();

    if du > surface.degree_u || dv > surface.degree_v {
        return Ok(Tensor::zeros(&[m, n_dims], DType::F64, device));
    }

    // Differentiate control points du times in u direction
    let mut diff_cp = surface.control_points.clone();
    let mut deg_u = surface.degree_u;
    let mut scale_u = 1.0;
    let mut cur_nu = nu;

    for _ in 0..du {
        // Forward difference along axis 0
        let hi = diff_cp.narrow(0, 1, cur_nu - 1)?.contiguous();
        let lo = diff_cp.narrow(0, 0, cur_nu - 1)?.contiguous();
        diff_cp = client.sub(&hi, &lo)?;
        scale_u *= deg_u as f64;
        deg_u -= 1;
        cur_nu -= 1;
    }

    // Differentiate dv times in v direction
    let mut deg_v = surface.degree_v;
    let mut scale_v = 1.0;
    let mut cur_nv = nv;

    for _ in 0..dv {
        let hi = diff_cp.narrow(1, 1, cur_nv - 1)?.contiguous();
        let lo = diff_cp.narrow(1, 0, cur_nv - 1)?.contiguous();
        diff_cp = client.sub(&hi, &lo)?;
        scale_v *= deg_v as f64;
        deg_v -= 1;
        cur_nv -= 1;
    }

    // Evaluate derivative surface
    let deriv_surface = BezierSurface {
        control_points: client.mul_scalar(&diff_cp, scale_u * scale_v)?,
        degree_u: deg_u,
        degree_v: deg_v,
    };

    bezier_surface_evaluate_impl(client, &deriv_surface, u, v)
}

/// Compute surface normals at (u, v) parameter pairs (fully on-device).
///
/// Normal = dS/du × dS/dv (cross product of partial derivatives).
pub fn bezier_surface_normal_impl<R, C>(
    client: &C,
    surface: &BezierSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    let n_dims = surface.control_points.shape()[2];
    if n_dims != 3 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "control_points".to_string(),
            message: "normals require 3D control points".to_string(),
        });
    }

    let du = bezier_surface_partial_impl(client, surface, u, v, 1, 0)?; // [m, 3]
    let dv = bezier_surface_partial_impl(client, surface, u, v, 0, 1)?; // [m, 3]

    // Cross product: du × dv
    cross_product_3d(client, &du, &dv)
}

/// Compute cross product of two [m, 3] tensors (fully on-device).
pub(crate) fn cross_product_3d<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let a0 = a.narrow(1, 0, 1)?.contiguous();
    let a1 = a.narrow(1, 1, 1)?.contiguous();
    let a2 = a.narrow(1, 2, 1)?.contiguous();
    let b0 = b.narrow(1, 0, 1)?.contiguous();
    let b1 = b.narrow(1, 1, 1)?.contiguous();
    let b2 = b.narrow(1, 2, 1)?.contiguous();

    let c0 = client.sub(&client.mul(&a1, &b2)?, &client.mul(&a2, &b1)?)?;
    let c1 = client.sub(&client.mul(&a2, &b0)?, &client.mul(&a0, &b2)?)?;
    let c2 = client.sub(&client.mul(&a0, &b1)?, &client.mul(&a1, &b0)?)?;

    let result = client.cat(&[&c0, &c1, &c2], 1)?;
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
    fn test_bezier_surface_bilinear() {
        let (device, client) = setup();
        // Bilinear patch: 4 corners in 3D
        // P00=(0,0,0), P01=(1,0,0), P10=(0,1,0), P11=(1,1,1)
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            &[2, 2, 3],
            &device,
        );
        let surface = BezierSurface {
            control_points: cp,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = bezier_surface_evaluate_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // At (0,0): (0,0,0)
        assert!((vals[0]).abs() < 1e-10);
        assert!((vals[1]).abs() < 1e-10);
        assert!((vals[2]).abs() < 1e-10);

        // At (0.5, 0.5): (0.5, 0.5, 0.25) — bilinear interpolation
        assert!((vals[3] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 0.5).abs() < 1e-10);
        assert!((vals[5] - 0.25).abs() < 1e-10);

        // At (1,1): (1,1,1)
        assert!((vals[6] - 1.0).abs() < 1e-10);
        assert!((vals[7] - 1.0).abs() < 1e-10);
        assert!((vals[8] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_surface_corners() {
        let (device, client) = setup();
        // Corners should interpolate control points at (0,0), (0,1), (1,0), (1,1)
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[2, 2, 3],
            &device,
        );
        let surface = BezierSurface {
            control_points: cp,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 0.0, 1.0], &[4], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let result = bezier_surface_evaluate_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // (0,0): P00 = (0,0,0)
        assert!((vals[0]).abs() < 1e-10);
        // (1,0): P10 = (0,1,0)
        assert!((vals[4] - 1.0).abs() < 1e-10);
        // (0,1): P01 = (1,0,0)
        assert!((vals[6] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_surface_normal() {
        let (device, client) = setup();
        // Flat patch in xy-plane: normals should point in z direction
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[2, 2, 3],
            &device,
        );
        let surface = BezierSurface {
            control_points: cp,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.5], &[1], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.5], &[1], &device);
        let normal = bezier_surface_normal_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = normal.to_vec();

        // Normal should be (0, 0, ±something) for flat xy patch
        assert!(vals[0].abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
        assert!(vals[2].abs() > 1e-10); // non-zero z component
    }
}
