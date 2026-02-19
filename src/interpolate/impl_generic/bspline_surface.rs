//! B-spline surface generic implementation (fully on-device).
//!
//! Uses tensor product of 1D Cox-de Boor bases for evaluation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bezier_surface::cross_product_3d;
use crate::interpolate::impl_generic::bspline::compute_basis_matrix;
use crate::interpolate::traits::bspline_surface::BSplineSurface;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validate B-spline surface parameters.
fn validate<R: Runtime<DType = DType>>(surface: &BSplineSurface<R>) -> InterpolateResult<()> {
    let shape = surface.control_points.shape();
    if shape.len() != 3 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "control_points".to_string(),
            message: "expected shape [nu, nv, n_dims]".to_string(),
        });
    }
    let nu = shape[0];
    let nv = shape[1];
    let nku = surface.knots_u.shape()[0];
    let nkv = surface.knots_v.shape()[0];

    if nku != nu + surface.degree_u + 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "knots_u".to_string(),
            message: format!(
                "expected {} knots_u, got {}",
                nu + surface.degree_u + 1,
                nku
            ),
        });
    }
    if nkv != nv + surface.degree_v + 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "knots_v".to_string(),
            message: format!(
                "expected {} knots_v, got {}",
                nv + surface.degree_v + 1,
                nkv
            ),
        });
    }
    Ok(())
}

/// Evaluate a B-spline surface at (u, v) parameter pairs (fully on-device).
pub fn bspline_surface_evaluate_impl<R, C>(
    client: &C,
    surface: &BSplineSurface<R>,
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
            context: "bspline_surface: u and v must have same length".to_string(),
        });
    }

    // B-spline bases
    let basis_u = compute_basis_matrix(client, u, &surface.knots_u, surface.degree_u, nu)?;
    let basis_v = compute_basis_matrix(client, v, &surface.knots_v, surface.degree_v, nv)?;

    // Tensor product: [m, nu*nv]
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

    let cp_flat = surface
        .control_points
        .reshape(&[nu * nv, n_dims])?
        .contiguous();
    let result = client.matmul(&product_flat, &cp_flat)?;
    Ok(result)
}

/// Evaluate partial derivatives of a B-spline surface (fully on-device).
///
/// Differentiates control points in u/v directions using forward differences
/// scaled by degree factors, then evaluates the reduced-degree surface.
pub fn bspline_surface_partial_impl<R, C>(
    client: &C,
    surface: &BSplineSurface<R>,
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
        return bspline_surface_evaluate_impl(client, surface, u, v);
    }

    validate(surface)?;

    let shape = surface.control_points.shape();
    let nu = shape[0];
    let nv = shape[1];
    let n_dims = shape[2];
    let m = u.shape()[0];
    let device = client.device();

    if du > surface.degree_u || dv > surface.degree_v {
        return Ok(Tensor::zeros(&[m, n_dims], DType::F64, device));
    }

    // Differentiate du times in u, dv times in v
    let mut diff_cp = surface.control_points.clone();
    let mut deg_u = surface.degree_u;
    let mut cur_nu = nu;
    let mut knots_u = surface.knots_u.clone();

    for _ in 0..du {
        let nku = knots_u.shape()[0];
        let hi = diff_cp.narrow(0, 1, cur_nu - 1)?.contiguous();
        let lo = diff_cp.narrow(0, 0, cur_nu - 1)?.contiguous();
        let delta = client.sub(&hi, &lo)?;

        // Scale by degree / (knot differences) per row
        // For B-spline derivative: scale factor = deg / (t_{i+deg+1} - t_{i+1})
        let t_hi = knots_u.narrow(0, deg_u + 1, cur_nu - 1)?.contiguous();
        let t_lo = knots_u.narrow(0, 1, cur_nu - 1)?.contiguous();
        let dt = client.sub(&t_hi, &t_lo)?;

        // Safe division
        let eps = Tensor::full_scalar(&[cur_nu - 1], DType::F64, 1e-300, device);
        let abs_dt = client.abs(&dt)?;
        let dt_safe = client.maximum(&abs_dt, &eps)?;
        let zero = Tensor::zeros(&[cur_nu - 1], DType::F64, device);
        let mask = client.gt(&abs_dt, &zero)?;

        // scale_factors: [cur_nu-1] → [cur_nu-1, 1, 1] for broadcasting
        let cur_nv = diff_cp.shape()[1];
        let scale = client.mul_scalar(
            &client.mul(
                &client.div(&Tensor::ones(&[cur_nu - 1], DType::F64, device), &dt_safe)?,
                &mask,
            )?,
            deg_u as f64,
        )?;
        let scale_broad = scale
            .reshape(&[cur_nu - 1, 1, 1])?
            .broadcast_to(&[cur_nu - 1, cur_nv, n_dims])?
            .contiguous();

        diff_cp = client.mul(&delta, &scale_broad)?;

        // Update knots: remove first and last
        knots_u = knots_u.narrow(0, 1, nku - 2)?.contiguous();
        deg_u -= 1;
        cur_nu -= 1;
    }

    let mut deg_v = surface.degree_v;
    let mut cur_nv = nv;
    let mut knots_v = surface.knots_v.clone();

    for _ in 0..dv {
        let nkv = knots_v.shape()[0];
        let hi = diff_cp.narrow(1, 1, cur_nv - 1)?.contiguous();
        let lo = diff_cp.narrow(1, 0, cur_nv - 1)?.contiguous();
        let delta = client.sub(&hi, &lo)?;

        let t_hi = knots_v.narrow(0, deg_v + 1, cur_nv - 1)?.contiguous();
        let t_lo = knots_v.narrow(0, 1, cur_nv - 1)?.contiguous();
        let dt = client.sub(&t_hi, &t_lo)?;

        let eps = Tensor::full_scalar(&[cur_nv - 1], DType::F64, 1e-300, device);
        let abs_dt = client.abs(&dt)?;
        let dt_safe = client.maximum(&abs_dt, &eps)?;
        let zero = Tensor::zeros(&[cur_nv - 1], DType::F64, device);
        let mask = client.gt(&abs_dt, &zero)?;

        let cur_nu_now = diff_cp.shape()[0];
        let scale = client.mul_scalar(
            &client.mul(
                &client.div(&Tensor::ones(&[cur_nv - 1], DType::F64, device), &dt_safe)?,
                &mask,
            )?,
            deg_v as f64,
        )?;
        let scale_broad = scale
            .reshape(&[1, cur_nv - 1, 1])?
            .broadcast_to(&[cur_nu_now, cur_nv - 1, n_dims])?
            .contiguous();

        diff_cp = client.mul(&delta, &scale_broad)?;

        knots_v = knots_v.narrow(0, 1, nkv - 2)?.contiguous();
        deg_v -= 1;
        cur_nv -= 1;
    }

    // Evaluate reduced surface
    let deriv_surface = BSplineSurface {
        control_points: diff_cp,
        knots_u,
        knots_v,
        degree_u: deg_u,
        degree_v: deg_v,
    };

    bspline_surface_evaluate_impl(client, &deriv_surface, u, v)
}

/// Compute surface normals at (u, v) parameter pairs (fully on-device).
pub fn bspline_surface_normal_impl<R, C>(
    client: &C,
    surface: &BSplineSurface<R>,
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

    let du = bspline_surface_partial_impl(client, surface, u, v, 1, 0)?;
    let dv = bspline_surface_partial_impl(client, surface, u, v, 0, 1)?;
    cross_product_3d(client, &du, &dv)
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
    fn test_bspline_surface_bilinear() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[2, 2, 3],
            &device,
        );
        let knots_u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let knots_v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let surface = BSplineSurface {
            control_points: cp,
            knots_u,
            knots_v,
            degree_u: 1,
            degree_v: 1,
        };

        let u = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = bspline_surface_evaluate_impl(&client, &surface, &u, &v).unwrap();
        let vals: Vec<f64> = result.to_vec();

        // At (0,0): (0,0,0)
        assert!((vals[0]).abs() < 1e-10);
        // At (0.5, 0.5): (0.5, 0.5, 0)
        assert!((vals[3] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 0.5).abs() < 1e-10);
        // At (1,1): (1,1,0)
        assert!((vals[6] - 1.0).abs() < 1e-10);
        assert!((vals[7] - 1.0).abs() < 1e-10);
    }
}
