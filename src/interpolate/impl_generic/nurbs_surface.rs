//! NURBS surface generic implementation (fully on-device).
//!
//! Uses rational extension of B-spline surface evaluation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bezier_surface::cross_product_3d;
use crate::interpolate::impl_generic::bspline::compute_basis_matrix;
use crate::interpolate::impl_generic::bspline_surface::{
    bspline_surface_evaluate_impl, bspline_surface_partial_impl,
};
use crate::interpolate::traits::bspline_surface::BSplineSurface;
use crate::interpolate::traits::nurbs_surface::NurbsSurface;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validate NURBS surface parameters.
fn validate<R: Runtime<DType = DType>>(surface: &NurbsSurface<R>) -> InterpolateResult<()> {
    let shape = surface.control_points.shape();
    if shape.len() != 3 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "control_points".to_string(),
            message: "expected shape [nu, nv, n_dims]".to_string(),
        });
    }
    let nu = shape[0];
    let nv = shape[1];
    let w_shape = surface.weights.shape();
    if w_shape.len() != 2 || w_shape[0] != nu || w_shape[1] != nv {
        return Err(InterpolateError::ShapeMismatch {
            expected: nu * nv,
            actual: w_shape.iter().product(),
            context: "nurbs_surface: weights must be [nu, nv]".to_string(),
        });
    }
    Ok(())
}

/// Evaluate a NURBS surface at (u, v) parameter pairs (fully on-device).
///
/// S(u,v) = sum_ij (w_ij * N_i(u) * N_j(v) * P_ij) / sum_ij (w_ij * N_i(u) * N_j(v))
pub fn nurbs_surface_evaluate_impl<R, C>(
    client: &C,
    surface: &NurbsSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate(surface)?;

    let shape = surface.control_points.shape();
    let nu = shape[0];
    let nv = shape[1];
    let n_dims = shape[2];
    let m = u.shape()[0];

    if m != v.shape()[0] {
        return Err(InterpolateError::ShapeMismatch {
            expected: m,
            actual: v.shape()[0],
            context: "nurbs_surface: u and v must have same length".to_string(),
        });
    }

    // B-spline bases
    let basis_u = compute_basis_matrix(client, u, &surface.knots_u, surface.degree_u, nu)?;
    let basis_v = compute_basis_matrix(client, v, &surface.knots_v, surface.degree_v, nv)?;

    // Tensor product basis: [m, nu*nv]
    let bu_exp = basis_u
        .reshape(&[m, nu, 1])?
        .broadcast_to(&[m, nu, nv])?
        .contiguous();
    let bv_exp = basis_v
        .reshape(&[m, 1, nv])?
        .broadcast_to(&[m, nu, nv])?
        .contiguous();
    let product = client.mul(&bu_exp, &bv_exp)?;

    // Weight the basis: [m, nu, nv] * [nu, nv] → [m, nu, nv]
    let w_broad = surface
        .weights
        .reshape(&[1, nu, nv])?
        .broadcast_to(&[m, nu, nv])?
        .contiguous();
    let weighted_basis = client.mul(&product, &w_broad)?;
    let wb_flat = weighted_basis.reshape(&[m, nu * nv])?;

    // Denominator: sum of weighted basis → [m, 1]
    let denominator = client.sum(&weighted_basis.reshape(&[m, nu * nv])?, &[1], true)?;

    // Weighted control points
    let w_cp_broad = surface
        .weights
        .reshape(&[nu, nv, 1])?
        .broadcast_to(&[nu, nv, n_dims])?
        .contiguous();
    let weighted_cp = client.mul(&surface.control_points, &w_cp_broad)?;
    let wcp_flat = weighted_cp.reshape(&[nu * nv, n_dims])?.contiguous();

    // Numerator: wb_flat @ wcp_flat → [m, n_dims]
    let numerator = client.matmul(&wb_flat, &wcp_flat)?;

    // Divide
    let denom_broad = denominator.broadcast_to(&[m, n_dims])?.contiguous();
    let result = client.div(&numerator, &denom_broad)?;
    Ok(result)
}

/// Evaluate partial derivatives of a NURBS surface (fully on-device).
///
/// Uses quotient rule on the rational form.
pub fn nurbs_surface_partial_impl<R, C>(
    client: &C,
    surface: &NurbsSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
    du: usize,
    dv: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if du == 0 && dv == 0 {
        return nurbs_surface_evaluate_impl(client, surface, u, v);
    }

    validate(surface)?;

    let shape = surface.control_points.shape();
    let nu = shape[0];
    let nv = shape[1];
    let n_dims = shape[2];
    let m = u.shape()[0];

    // Build homogeneous B-spline surface: weighted CP and weight function
    let w_cp_broad = surface
        .weights
        .reshape(&[nu, nv, 1])?
        .broadcast_to(&[nu, nv, n_dims])?
        .contiguous();
    let weighted_cp = client.mul(&surface.control_points, &w_cp_broad)?;

    // Numerator surface (A): weighted control points
    let a_surface = BSplineSurface {
        control_points: weighted_cp,
        knots_u: surface.knots_u.clone(),
        knots_v: surface.knots_v.clone(),
        degree_u: surface.degree_u,
        degree_v: surface.degree_v,
    };

    // Weight surface (w): weights as [nu, nv, 1]
    let w_surface = BSplineSurface {
        control_points: surface.weights.reshape(&[nu, nv, 1])?,
        knots_u: surface.knots_u.clone(),
        knots_v: surface.knots_v.clone(),
        degree_u: surface.degree_u,
        degree_v: surface.degree_v,
    };

    if du + dv == 1 {
        // First partial derivative via quotient rule: (A' * w - A * w') / w^2
        let a_val = bspline_surface_evaluate_impl(client, &a_surface, u, v)?;
        let a_deriv = bspline_surface_partial_impl(client, &a_surface, u, v, du, dv)?;
        let w_val = bspline_surface_evaluate_impl(client, &w_surface, u, v)?; // [m, 1]
        let w_deriv = bspline_surface_partial_impl(client, &w_surface, u, v, du, dv)?;

        let w_broad2 = w_val.broadcast_to(&[m, n_dims])?.contiguous();
        let wd_broad = w_deriv.broadcast_to(&[m, n_dims])?.contiguous();

        let num = client.sub(
            &client.mul(&a_deriv, &w_broad2)?,
            &client.mul(&a_val, &wd_broad)?,
        )?;
        let w_sq = client.mul(&w_broad2, &w_broad2)?;
        Ok(client.div(&num, &w_sq)?)
    } else {
        // Higher order: numerical differentiation
        let device = client.device();
        let h = 1e-7;

        if du > 0 {
            let h_t = Tensor::full_scalar(&[m], DType::F64, h, device);
            let u_plus = client.add(u, &h_t)?;
            let u_minus = client.sub(u, &h_t)?;
            let f_plus = nurbs_surface_partial_impl(client, surface, &u_plus, v, du - 1, dv)?;
            let f_minus = nurbs_surface_partial_impl(client, surface, &u_minus, v, du - 1, dv)?;
            Ok(client.mul_scalar(&client.sub(&f_plus, &f_minus)?, 0.5 / h)?)
        } else {
            let h_t = Tensor::full_scalar(&[m], DType::F64, h, device);
            let v_plus = client.add(v, &h_t)?;
            let v_minus = client.sub(v, &h_t)?;
            let f_plus = nurbs_surface_partial_impl(client, surface, u, &v_plus, du, dv - 1)?;
            let f_minus = nurbs_surface_partial_impl(client, surface, u, &v_minus, du, dv - 1)?;
            Ok(client.mul_scalar(&client.sub(&f_plus, &f_minus)?, 0.5 / h)?)
        }
    }
}

/// Compute surface normals at (u, v) parameter pairs (fully on-device).
pub fn nurbs_surface_normal_impl<R, C>(
    client: &C,
    surface: &NurbsSurface<R>,
    u: &Tensor<R>,
    v: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n_dims = surface.control_points.shape()[2];
    if n_dims != 3 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "control_points".to_string(),
            message: "normals require 3D control points".to_string(),
        });
    }

    let du_val = nurbs_surface_partial_impl(client, surface, u, v, 1, 0)?;
    let dv_val = nurbs_surface_partial_impl(client, surface, u, v, 0, 1)?;
    cross_product_3d(client, &du_val, &dv_val)
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
    fn test_nurbs_surface_uniform_weights() {
        let (device, client) = setup();
        // Uniform weights → same as B-spline surface
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[2, 2, 3],
            &device,
        );
        let weights = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], &device);
        let knots_u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let knots_v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let surface = NurbsSurface {
            control_points: cp,
            weights,
            knots_u,
            knots_v,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = nurbs_surface_evaluate_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // At (0.5, 0.5): (0.5, 0.5, 0)
        assert!((vals[3] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 0.5).abs() < 1e-10);
        assert!((vals[5]).abs() < 1e-10);
    }

    #[test]
    fn test_nurbs_surface_corners() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[2, 2, 2],
            &device,
        );
        let weights = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], &device);
        let knots_u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let knots_v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let surface = NurbsSurface {
            control_points: cp,
            weights,
            knots_u,
            knots_v,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let result = nurbs_surface_evaluate_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = result.to_vec();

        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.0).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }
}
