//! NURBS curve generic implementation (fully on-device).
//!
//! Uses B-spline basis evaluation and rational weighting.
//! C(t) = sum(w_i * N_i(t) * P_i) / sum(w_i * N_i(t))

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bspline::compute_basis_matrix;
use crate::interpolate::impl_generic::bspline_curve::{
    bspline_curve_derivative_impl, bspline_curve_evaluate_impl,
};
use crate::interpolate::traits::bspline_curve::BSplineCurve;
use crate::interpolate::traits::nurbs_curve::NurbsCurve;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validate NURBS curve parameters.
fn validate_nurbs_curve<R: Runtime>(curve: &NurbsCurve<R>) -> InterpolateResult<()> {
    let n_points = curve.control_points.shape()[0];
    let n_weights = curve.weights.shape()[0];
    let n_knots = curve.knots.shape()[0];
    let expected_knots = n_points + curve.degree + 1;

    if n_weights != n_points {
        return Err(InterpolateError::ShapeMismatch {
            expected: n_points,
            actual: n_weights,
            context: "nurbs_curve: weights must match number of control points".to_string(),
        });
    }

    if n_knots != expected_knots {
        return Err(InterpolateError::InvalidParameter {
            parameter: "knots".to_string(),
            message: format!(
                "expected {} knots for {} control points with degree {}, got {}",
                expected_knots, n_points, curve.degree, n_knots
            ),
        });
    }

    Ok(())
}

/// Evaluate a NURBS curve at parameter values t (fully on-device).
///
/// C(t) = (basis @ (W * P)) / (basis @ W)
/// where W = diag(weights), P = control_points
pub fn nurbs_curve_evaluate_impl<R, C>(
    client: &C,
    curve: &NurbsCurve<R>,
    t: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate_nurbs_curve(curve)?;

    let n_points = curve.control_points.shape()[0];
    let n_dims = curve.control_points.shape()[1];
    let m = t.shape()[0];

    // Compute B-spline basis [m, n_points]
    let basis = compute_basis_matrix(client, t, &curve.knots, curve.degree, n_points)?;

    // Weighted control points: w_i * P_i for each point
    // weights [n_points] → [n_points, 1], broadcast to [n_points, n_dims]
    let w_col = curve.weights.reshape(&[n_points, 1])?;
    let w_broad = w_col.broadcast_to(&[n_points, n_dims])?.contiguous();
    let weighted_cp = client.mul(&curve.control_points, &w_broad)?; // [n_points, n_dims]

    // Numerator: basis @ weighted_cp → [m, n_dims]
    let numerator = client.matmul(&basis, &weighted_cp)?;

    // Denominator: basis @ weights → [m, 1]
    let w_column = curve.weights.reshape(&[n_points, 1])?;
    let denominator = client.matmul(&basis, &w_column)?; // [m, 1]

    // Result = numerator / denominator (broadcast denominator)
    let denom_broad = denominator.broadcast_to(&[m, n_dims])?.contiguous();
    let result = client.div(&numerator, &denom_broad)?;
    Ok(result)
}

/// Evaluate the derivative of a NURBS curve at parameter values t (fully on-device).
///
/// Uses the quotient rule: d/dt [A(t)/w(t)] = (A'(t)*w(t) - A(t)*w'(t)) / w(t)^2
/// where A(t) = sum(w_i * N_i(t) * P_i) and w(t) = sum(w_i * N_i(t))
pub fn nurbs_curve_derivative_impl<R, C>(
    client: &C,
    curve: &NurbsCurve<R>,
    t: &Tensor<R>,
    order: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if order == 0 {
        return nurbs_curve_evaluate_impl(client, curve, t);
    }

    validate_nurbs_curve(curve)?;

    let n_points = curve.control_points.shape()[0];
    let n_dims = curve.control_points.shape()[1];
    let m = t.shape()[0];

    // Build weighted B-spline curve for the numerator: A(t) = sum(w_i * N_i(t) * P_i)
    let w_col = curve.weights.reshape(&[n_points, 1])?;
    let w_broad = w_col.broadcast_to(&[n_points, n_dims])?.contiguous();
    let weighted_cp = client.mul(&curve.control_points, &w_broad)?;

    let a_curve = BSplineCurve {
        control_points: weighted_cp,
        knots: curve.knots.clone(),
        degree: curve.degree,
    };

    // Build weight function as 1D B-spline: w(t) = sum(w_i * N_i(t))
    let w_curve = BSplineCurve {
        control_points: curve.weights.reshape(&[n_points, 1])?,
        knots: curve.knots.clone(),
        degree: curve.degree,
    };

    if order == 1 {
        // First derivative: (A' * w - A * w') / w^2
        let a_val = bspline_curve_evaluate_impl(client, &a_curve, t)?; // [m, n_dims]
        let a_deriv = bspline_curve_derivative_impl(client, &a_curve, t, 1)?; // [m, n_dims]
        let w_val = bspline_curve_evaluate_impl(client, &w_curve, t)?; // [m, 1]
        let w_deriv = bspline_curve_derivative_impl(client, &w_curve, t, 1)?; // [m, 1]

        let w_broad2 = w_val.broadcast_to(&[m, n_dims])?.contiguous();
        let wd_broad = w_deriv.broadcast_to(&[m, n_dims])?.contiguous();

        let num = client.sub(
            &client.mul(&a_deriv, &w_broad2)?,
            &client.mul(&a_val, &wd_broad)?,
        )?;
        let w_sq = client.mul(&w_broad2, &w_broad2)?;
        let result = client.div(&num, &w_sq)?;
        Ok(result)
    } else {
        // Higher order: recursive via quotient rule generalization
        // For simplicity, compute numerically via finite differences on the first derivative
        let device = client.device();
        let h = 1e-7;
        let h_tensor = Tensor::full_scalar(&[m], DType::F64, h, device);
        let t_plus = client.add(t, &h_tensor)?;
        let t_minus = client.sub(t, &h_tensor)?;

        let d_plus = nurbs_curve_derivative_impl(client, curve, &t_plus, order - 1)?;
        let d_minus = nurbs_curve_derivative_impl(client, curve, &t_minus, order - 1)?;

        let result = client.mul_scalar(&client.sub(&d_plus, &d_minus)?, 0.5 / h)?;
        Ok(result)
    }
}

/// Subdivide a NURBS curve at parameter t (fully on-device).
///
/// Converts to homogeneous coordinates (w*P, w), subdivides the B-spline,
/// then converts back.
pub fn nurbs_curve_subdivide_impl<R, C>(
    client: &C,
    curve: &NurbsCurve<R>,
    t: f64,
) -> InterpolateResult<(NurbsCurve<R>, NurbsCurve<R>)>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate_nurbs_curve(curve)?;

    let n_points = curve.control_points.shape()[0];
    let n_dims = curve.control_points.shape()[1];

    // Build homogeneous coordinates: [w*P | w] → [n_points, n_dims+1]
    let w_col = curve.weights.reshape(&[n_points, 1])?;
    let w_broad = w_col.broadcast_to(&[n_points, n_dims])?.contiguous();
    let weighted_cp = client.mul(&curve.control_points, &w_broad)?;
    let homo_cp = client.cat(&[&weighted_cp, &w_col], 1)?; // [n_points, n_dims+1]

    let homo_curve = BSplineCurve {
        control_points: homo_cp,
        knots: curve.knots.clone(),
        degree: curve.degree,
    };

    // Subdivide in homogeneous space
    let (left_homo, right_homo) =
        crate::interpolate::impl_generic::bspline_curve::bspline_curve_subdivide_impl(
            client,
            &homo_curve,
            t,
        )?;

    // Convert back from homogeneous coordinates
    let left = dehomogenize(client, &left_homo, n_dims)?;
    let right = dehomogenize(client, &right_homo, n_dims)?;

    Ok((left, right))
}

/// Convert a homogeneous B-spline curve back to NURBS.
fn dehomogenize<R, C>(
    client: &C,
    homo_curve: &BSplineCurve<R>,
    n_dims: usize,
) -> InterpolateResult<NurbsCurve<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n_points = homo_curve.control_points.shape()[0];

    // Extract weights (last column)
    let weights = homo_curve
        .control_points
        .narrow(1, n_dims, 1)?
        .contiguous()
        .reshape(&[n_points])?;

    // Extract weighted coordinates and divide by weights
    let weighted_cp = homo_curve.control_points.narrow(1, 0, n_dims)?.contiguous();
    let w_col = weights.reshape(&[n_points, 1])?;
    let w_broad = w_col.broadcast_to(&[n_points, n_dims])?.contiguous();
    let control_points = client.div(&weighted_cp, &w_broad)?;

    Ok(NurbsCurve {
        control_points,
        weights,
        knots: homo_curve.knots.clone(),
        degree: homo_curve.degree,
    })
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
    fn test_nurbs_uniform_weights_equals_bspline() {
        let (device, client) = setup();
        // With uniform weights, NURBS should equal B-spline
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[2, 2], &device);
        let weights = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device);
        let knots = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let nurbs = NurbsCurve {
            control_points: cp.clone(),
            weights,
            knots: knots.clone(),
            degree: 1,
        };
        let bspline = BSplineCurve {
            control_points: cp,
            knots,
            degree: 1,
        };

        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], &device);
        let nurbs_result = nurbs_curve_evaluate_impl(&client, &nurbs, &t).unwrap();
        let bspline_result = bspline_curve_evaluate_impl(&client, &bspline, &t).unwrap();

        let nv: Vec<f64> = nurbs_result.to_vec();
        let bv: Vec<f64> = bspline_result.to_vec();
        for (a, b) in nv.iter().zip(bv.iter()) {
            assert!((a - b).abs() < 1e-10, "NURBS {} != B-spline {}", a, b);
        }
    }

    #[test]
    fn test_nurbs_circle_arc() {
        let (device, client) = setup();
        // Quarter circle: 3 control points with weight sqrt(2)/2 for middle
        let cp =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], &[3, 2], &device);
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let weights = Tensor::<CpuRuntime>::from_slice(&[1.0, w, 1.0], &[3], &device);
        let knots =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6], &device);

        let nurbs = NurbsCurve {
            control_points: cp,
            weights,
            knots,
            degree: 2,
        };

        // Points on the circle should have |p| = 1
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], &device);
        let result = nurbs_curve_evaluate_impl(&client, &nurbs, &t).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for i in 0..5 {
            let x = vals[i * 2];
            let y = vals[i * 2 + 1];
            let r = (x * x + y * y).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-8,
                "point {} ({}, {}) has radius {} != 1",
                i,
                x,
                y,
                r
            );
        }
    }

    #[test]
    fn test_nurbs_derivative_linear() {
        let (device, client) = setup();
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 2.0, 4.0], &[2, 2], &device);
        let weights = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device);
        let knots = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);

        let nurbs = NurbsCurve {
            control_points: cp,
            weights,
            knots,
            degree: 1,
        };

        let t = Tensor::<CpuRuntime>::from_slice(&[0.25, 0.75], &[2], &device);
        let deriv = nurbs_curve_derivative_impl(&client, &nurbs, &t, 1).unwrap();
        let vals: Vec<f64> = deriv.to_vec();
        // With uniform weights, derivative = (P1-P0) / (t_end - t_start) * degree... = (2,4)
        assert!((vals[0] - 2.0).abs() < 1e-6, "dx={}", vals[0]);
        assert!((vals[1] - 4.0).abs() < 1e-6, "dy={}", vals[1]);
    }
}
