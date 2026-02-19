//! Bezier curve generic implementation (fully on-device).
//!
//! Uses Bernstein polynomial matrix method for GPU-friendly evaluation.
//! Binomial coefficients computed via lgamma for numerical stability.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::bezier_curve::BezierCurve;
use numr::algorithm::special::SpecialFunctions;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute Bernstein basis matrix [m, n+1] for degree n at parameter values t.
///
/// B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
/// Binomial coefficients via: exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
pub(crate) fn bernstein_basis_matrix<R, C>(
    client: &C,
    t: &Tensor<R>,
    degree: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    let device = client.device();
    let m = t.shape()[0];
    let n = degree;
    let n_basis = n + 1;

    if n == 0 {
        return Ok(Tensor::ones(&[m, 1], DType::F64, device));
    }

    // t_col: [m, 1], one_minus_t: [m, 1]
    let t_col = t.reshape(&[m, 1])?;
    let one = Tensor::ones(&[m, 1], DType::F64, device);
    let one_minus_t = client.sub(&one, &t_col)?;

    // i_vals: [1, n+1] = [0, 1, ..., n]
    let i_vals = client.arange(0.0, n_basis as f64, 1.0, DType::F64)?;
    let i_row = i_vals.reshape(&[1, n_basis])?;

    // Binomial coefficients: C(n, i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
    let n_plus_1 = Tensor::full_scalar(&[1, n_basis], DType::F64, (n + 1) as f64, device);
    let one_1n = Tensor::ones(&[1, n_basis], DType::F64, device);
    let i_plus_1 = client.add(&i_row, &one_1n)?;
    let n_minus_i_plus_1 = client.sub(&n_plus_1, &i_row)?;

    let lg_n1 = client.lgamma(&n_plus_1)?;
    let lg_i1 = client.lgamma(&i_plus_1)?;
    let lg_ni1 = client.lgamma(&n_minus_i_plus_1)?;

    let log_binom = client.sub(&lg_n1, &client.add(&lg_i1, &lg_ni1)?)?;
    let binom = client.exp(&log_binom)?; // [1, n+1]

    // t^i: broadcast t [m,1] and i [1,n+1] → [m, n+1]
    let t_broad = t_col.broadcast_to(&[m, n_basis])?.contiguous();
    let i_broad = i_row.broadcast_to(&[m, n_basis])?.contiguous();
    let t_pow_i = client.pow(&t_broad, &i_broad)?;

    // (1-t)^(n-i): exponent = n - i
    let n_tensor = Tensor::full_scalar(&[1, n_basis], DType::F64, n as f64, device);
    let n_minus_i = client.sub(&n_tensor, &i_row)?;
    let omt_broad = one_minus_t.broadcast_to(&[m, n_basis])?.contiguous();
    let nmi_broad = n_minus_i.broadcast_to(&[m, n_basis])?.contiguous();
    let omt_pow = client.pow(&omt_broad, &nmi_broad)?;

    // basis = binom * t^i * (1-t)^(n-i)
    let binom_broad = binom.broadcast_to(&[m, n_basis])?.contiguous();
    let basis = client.mul(&binom_broad, &client.mul(&t_pow_i, &omt_pow)?)?;

    Ok(basis)
}

/// Evaluate a Bezier curve at parameter values t (fully on-device).
///
/// result = bernstein_basis(t, degree) @ control_points
pub fn bezier_evaluate_impl<R, C>(
    client: &C,
    curve: &BezierCurve<R>,
    t: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    let n_points = curve.control_points.shape()[0];
    if n_points != curve.degree + 1 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "degree".to_string(),
            message: format!(
                "degree {} requires {} control points, got {}",
                curve.degree,
                curve.degree + 1,
                n_points
            ),
        });
    }

    let basis = bernstein_basis_matrix(client, t, curve.degree)?; // [m, n+1]
    let result = client.matmul(&basis, &curve.control_points)?; // [m, n_dims]
    Ok(result)
}

/// Evaluate the derivative of a Bezier curve at parameter values t (fully on-device).
///
/// The k-th derivative of a degree-n Bezier curve is a degree-(n-k) Bezier curve
/// with control points computed from forward differences of the original control points.
pub fn bezier_derivative_impl<R, C>(
    client: &C,
    curve: &BezierCurve<R>,
    t: &Tensor<R>,
    order: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    if order == 0 {
        return bezier_evaluate_impl(client, curve, t);
    }

    let n = curve.degree;
    if order > n {
        let m = t.shape()[0];
        let n_dims = curve.control_points.shape()[1];
        let device = client.device();
        return Ok(Tensor::zeros(&[m, n_dims], DType::F64, device));
    }

    // Compute derivative control points via forward differences.
    // d^1 P_i = n * (P_{i+1} - P_i)
    // d^k P_i = n*(n-1)*...*(n-k+1) * Delta^k P_i
    let mut diff_points = curve.control_points.clone();
    let mut current_n = n;
    let mut scale = 1.0;

    for _ in 0..order {
        let n_pts = diff_points.shape()[0];
        let hi = diff_points.narrow(0, 1, n_pts - 1)?.contiguous();
        let lo = diff_points.narrow(0, 0, n_pts - 1)?.contiguous();
        diff_points = client.sub(&hi, &lo)?;
        scale *= current_n as f64;
        current_n -= 1;
    }

    // Derivative curve is degree (n - order) Bezier with scaled control points
    let deriv_points = client.mul_scalar(&diff_points, scale)?;
    let deriv_curve = BezierCurve {
        control_points: deriv_points,
        degree: n - order,
    };

    bezier_evaluate_impl(client, &deriv_curve, t)
}

/// Subdivide a Bezier curve at parameter t using de Casteljau's algorithm (fully on-device).
///
/// Performs n iterations of linear interpolation. Left curve collects first points
/// of each level, right curve collects last points (reversed).
pub fn bezier_subdivide_impl<R, C>(
    client: &C,
    curve: &BezierCurve<R>,
    t: f64,
) -> InterpolateResult<(BezierCurve<R>, BezierCurve<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
{
    let n = curve.degree;

    if n == 0 {
        return Ok((curve.clone(), curve.clone()));
    }

    // De Casteljau: iteratively lerp adjacent points
    // left[0] = P_0, left[i] = first point at level i
    // right[n] = P_n, right[n-i] = last point at level i
    let mut current = curve.control_points.clone();
    let mut left_points = Vec::with_capacity(n + 1);
    let mut right_points = Vec::with_capacity(n + 1);

    // First point of level 0
    left_points.push(current.narrow(0, 0, 1)?.contiguous());
    // Last point of level 0
    right_points.push(current.narrow(0, n, 1)?.contiguous());

    for _ in 0..n {
        let n_pts = current.shape()[0];
        let lo = current.narrow(0, 0, n_pts - 1)?.contiguous();
        let hi = current.narrow(0, 1, n_pts - 1)?.contiguous();
        // lerp: (1-t)*lo + t*hi
        let lo_part = client.mul_scalar(&lo, 1.0 - t)?;
        let hi_part = client.mul_scalar(&hi, t)?;
        current = client.add(&lo_part, &hi_part)?;

        left_points.push(current.narrow(0, 0, 1)?.contiguous());
        right_points.push(current.narrow(0, current.shape()[0] - 1, 1)?.contiguous());
    }

    // Build left control points: cat all left_points
    let left_refs: Vec<&Tensor<R>> = left_points.iter().collect();
    let left_cp = client.cat(&left_refs, 0)?;

    // Build right control points: reverse right_points
    right_points.reverse();
    let right_refs: Vec<&Tensor<R>> = right_points.iter().collect();
    let right_cp = client.cat(&right_refs, 0)?;

    Ok((
        BezierCurve {
            control_points: left_cp,
            degree: n,
        },
        BezierCurve {
            control_points: right_cp,
            degree: n,
        },
    ))
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
    fn test_bezier_linear() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[2, 2], &device);
        let curve = BezierCurve {
            control_points: cp,
            degree: 1,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = bezier_evaluate_impl(&client, &curve, &t).unwrap();
        let vals: Vec<f64> = result.to_vec();
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.0).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_quadratic() {
        let (device, client) = setup();
        // Quadratic Bezier: P0=(0,0), P1=(0.5,1), P2=(1,0)
        let cp =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.5, 1.0, 1.0, 0.0], &[3, 2], &device);
        let curve = BezierCurve {
            control_points: cp,
            degree: 2,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = bezier_evaluate_impl(&client, &curve, &t).unwrap();
        let vals: Vec<f64> = result.to_vec();
        // At t=0: (0,0)
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.0).abs() < 1e-10);
        // At t=0.5: (0.5, 0.5)
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - 0.5).abs() < 1e-10);
        // At t=1: (1,0)
        assert!((vals[4] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_derivative_linear() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 2.0, 4.0], &[2, 2], &device);
        let curve = BezierCurve {
            control_points: cp,
            degree: 1,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let deriv = bezier_derivative_impl(&client, &curve, &t, 1).unwrap();
        let vals: Vec<f64> = deriv.to_vec();
        // Derivative of linear Bezier is constant: (2, 4)
        for i in 0..3 {
            assert!((vals[i * 2] - 2.0).abs() < 1e-10);
            assert!((vals[i * 2 + 1] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bezier_subdivide() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[2, 2], &device);
        let curve = BezierCurve {
            control_points: cp,
            degree: 1,
        };
        let (left, right) = bezier_subdivide_impl(&client, &curve, 0.5).unwrap();

        // Evaluate both halves at their midpoints
        let t_mid = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let left_end = bezier_evaluate_impl(&client, &left, &t_mid).unwrap();
        let right_start =
            bezier_evaluate_impl(&client, &right, &Tensor::from_slice(&[0.0], &[1], &device))
                .unwrap();
        let lv: Vec<f64> = left_end.to_vec();
        let rv: Vec<f64> = right_start.to_vec();
        // Both should meet at the split point (0.5, 0.5)
        assert!((lv[0] - 0.5).abs() < 1e-10);
        assert!((lv[1] - 0.5).abs() < 1e-10);
        assert!((rv[0] - 0.5).abs() < 1e-10);
        assert!((rv[1] - 0.5).abs() < 1e-10);
    }
}
