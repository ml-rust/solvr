//! B-spline curve generic implementation (fully on-device).
//!
//! Reuses `compute_basis_matrix` from the existing B-spline interpolation module
//! for Cox-de Boor basis evaluation.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::impl_generic::bspline::{
    compute_basis_matrix, differentiate_bspline_tensor,
};
use crate::interpolate::traits::bspline::BSpline;
use crate::interpolate::traits::bspline_curve::BSplineCurve;
use numr::ops::{CompareOps, ConditionalOps, ReduceOps, ScalarOps, SortingOps, TypeConversionOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Validate B-spline curve parameters.
fn validate_bspline_curve<R: Runtime<DType = DType>>(
    curve: &BSplineCurve<R>,
) -> InterpolateResult<()> {
    let n_points = curve.control_points.shape()[0];
    let n_knots = curve.knots.shape()[0];
    let expected_knots = n_points + curve.degree + 1;

    if n_knots != expected_knots {
        return Err(InterpolateError::InvalidParameter {
            parameter: "knots".to_string(),
            message: format!(
                "expected {} knots for {} control points with degree {}, got {}",
                expected_knots, n_points, curve.degree, n_knots
            ),
        });
    }

    if n_points < curve.degree + 1 {
        return Err(InterpolateError::InsufficientData {
            required: curve.degree + 1,
            actual: n_points,
            context: "bspline_curve: need at least degree+1 control points".to_string(),
        });
    }

    Ok(())
}

/// Evaluate a B-spline curve at parameter values t (fully on-device).
///
/// Builds basis matrix via Cox-de Boor recurrence, then matmul with control points.
pub fn bspline_curve_evaluate_impl<R, C>(
    client: &C,
    curve: &BSplineCurve<R>,
    t: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate_bspline_curve(curve)?;
    let n_points = curve.control_points.shape()[0];

    // Compute basis matrix [m, n_points] using existing Cox-de Boor
    let basis = compute_basis_matrix(client, t, &curve.knots, curve.degree, n_points)?;

    // result = basis @ control_points → [m, n_dims]
    let result = client.matmul(&basis, &curve.control_points)?;
    Ok(result)
}

/// Evaluate the derivative of a B-spline curve at parameter values t (fully on-device).
///
/// Uses the knot differencing formula to compute derivative control points,
/// then evaluates the resulting lower-degree B-spline curve.
pub fn bspline_curve_derivative_impl<R, C>(
    client: &C,
    curve: &BSplineCurve<R>,
    t: &Tensor<R>,
    order: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if order == 0 {
        return bspline_curve_evaluate_impl(client, curve, t);
    }

    validate_bspline_curve(curve)?;

    if order > curve.degree {
        let m = t.shape()[0];
        let n_dims = curve.control_points.shape()[1];
        let device = client.device();
        return Ok(Tensor::zeros(&[m, n_dims], DType::F64, device));
    }

    // Apply derivative formula per dimension using the 1D differentiate_bspline_tensor.
    // We differentiate a reference 1D spline first to get the derivative knots and degree
    // (these are the same for all dimensions), then compute the basis matrix once.
    let n_points = curve.control_points.shape()[0];
    let n_dims = curve.control_points.shape()[1];

    // Build a reference 1D spline (dimension 0) to get derivative knots/degree
    let coeffs_0 = curve.control_points.narrow(1, 0, 1)?.contiguous();
    let coeffs_0_1d = coeffs_0.reshape(&[n_points])?;
    let ref_spline = BSpline {
        knots: curve.knots.clone(),
        coefficients: coeffs_0_1d,
        degree: curve.degree,
    };
    let ref_deriv = differentiate_bspline_tensor(client, &ref_spline, order)?;
    let n_coeffs = ref_deriv.coefficients.shape()[0];

    // Compute basis matrix ONCE (shared across all dimensions)
    let basis = compute_basis_matrix(client, t, &ref_deriv.knots, ref_deriv.degree, n_coeffs)?;

    // Collect derivative coefficients for all dimensions
    let mut dim_results = Vec::with_capacity(n_dims);
    // First dimension already computed
    let coeffs_col = ref_deriv.coefficients.reshape(&[n_coeffs, 1])?;
    dim_results.push(client.matmul(&basis, &coeffs_col)?);

    // Remaining dimensions reuse the same derivative knots/degree
    for d in 1..n_dims {
        let coeffs_d = curve.control_points.narrow(1, d, 1)?.contiguous();
        let coeffs_1d = coeffs_d.reshape(&[n_points])?;

        let spline_1d = BSpline {
            knots: curve.knots.clone(),
            coefficients: coeffs_1d,
            degree: curve.degree,
        };

        let deriv_spline = differentiate_bspline_tensor(client, &spline_1d, order)?;
        let deriv_col = deriv_spline.coefficients.reshape(&[n_coeffs, 1])?;
        dim_results.push(client.matmul(&basis, &deriv_col)?);
    }

    let refs: Vec<&Tensor<R>> = dim_results.iter().collect();
    let result = client.cat(&refs, 1)?; // [m, n_dims]
    Ok(result)
}

/// Subdivide a B-spline curve at parameter t via Boehm knot insertion (fully on-device).
///
/// Inserts knot t enough times (degree times) to split the curve into two.
/// Uses `searchsorted` for on-device knot span finding and tensor ops for alpha blending.
pub fn bspline_curve_subdivide_impl<R, C>(
    client: &C,
    curve: &BSplineCurve<R>,
    t: f64,
) -> InterpolateResult<(BSplineCurve<R>, BSplineCurve<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + SortingOps<R>
        + ReduceOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_bspline_curve(curve)?;

    let device = client.device();
    let n_points = curve.control_points.shape()[0];
    let k = curve.degree;
    let n_knots = curve.knots.shape()[0];

    // Find knot span on-device using searchsorted.
    // searchsorted(knots, [t], right=true) returns the first index where knots > t.
    // span = clamp(idx - 1, k, n_points - 1)
    let t_tensor = Tensor::from_slice(&[t], &[1], device);
    let idx_tensor = client.searchsorted(&curve.knots, &t_tensor, true)?;
    // Single scalar extraction for control flow index
    let raw_idx: i64 = idx_tensor.item()?;
    let span = (raw_idx - 1).max(k as i64).min((n_points - 1) as i64) as usize;

    // Count existing multiplicity of t on-device:
    // |knots - t| < eps, cast to F64, sum
    let t_broadcast = Tensor::from_slice(&[t], &[1], device).broadcast_to(&[n_knots])?;
    let diff = client.abs(&client.sub(&curve.knots, &t_broadcast)?)?;
    let eps_tensor = Tensor::from_slice(&[1e-15], &[1], device).broadcast_to(&[n_knots])?;
    let near_mask = client.lt(&diff, &eps_tensor)?; // boolean/i64 mask
    let near_f64 = client.cast(&near_mask, DType::F64)?;
    let mult_tensor = client.sum(&near_f64, &[0], false)?;
    // Single scalar for control flow (acceptable)
    let mult_val: f64 = mult_tensor.item()?;
    let mult = mult_val as usize;

    let insertions = k.saturating_sub(mult);
    if insertions == 0 {
        // Already fully split at this knot — just split by index
        let left_cp = curve.control_points.narrow(0, 0, span + 1)?.contiguous();
        let right_cp = curve
            .control_points
            .narrow(0, span, n_points - span)?
            .contiguous();

        let left_knots = curve.knots.narrow(0, 0, span + k + 2)?.contiguous();
        let right_knots = curve.knots.narrow(0, span, n_knots - span)?.contiguous();

        return Ok((
            BSplineCurve {
                control_points: left_cp,
                knots: left_knots,
                degree: k,
            },
            BSplineCurve {
                control_points: right_cp,
                knots: right_knots,
                degree: k,
            },
        ));
    }

    // Boehm knot insertion: insert t one at a time.
    // Each insertion is a small, bounded operation (at most k+1 affected points).
    // We keep all data on-device, computing alpha blending weights via tensor ops.
    let mut cp = curve.control_points.clone();
    let mut knots = curve.knots.clone();
    let mut current_span = span;

    for _ in 0..insertions {
        let n_cp = cp.shape()[0];
        let n_kn = knots.shape()[0];

        // Insert new knot value into knot vector
        let left_knots = knots.narrow(0, 0, current_span + 1)?.contiguous();
        let t_knot = Tensor::from_slice(&[t], &[1], device);
        let right_knots = knots
            .narrow(0, current_span + 1, n_kn - current_span - 1)?
            .contiguous();
        knots = client.cat(&[&left_knots, &t_knot, &right_knots], 0)?;

        // Compute blending alphas for affected range using tensor ops.
        // Affected range: [start..=current_span]
        let start = if current_span >= k {
            current_span - k + 1
        } else {
            0
        };
        let n_affected = current_span - start + 1;

        // knots_lo = knots[start..start+n_affected], knots_hi = knots[start+k+1..start+k+1+n_affected]
        let knots_lo = knots.narrow(0, start, n_affected)?.contiguous();
        let knots_hi = knots.narrow(0, start + k + 1, n_affected)?.contiguous();
        let denom = client.sub(&knots_hi, &knots_lo)?;

        let t_bcast = Tensor::from_slice(&[t], &[1], device).broadcast_to(&[n_affected])?;
        let numer = client.sub(&t_bcast, &knots_lo)?;

        // Safe division: where denom ~= 0, alpha = 0
        let eps_bcast = Tensor::from_slice(&[1e-300], &[1], device).broadcast_to(&[n_affected])?;
        let safe_denom = client.where_cond(
            &client.lt(&client.abs(&denom)?, &eps_bcast)?,
            &Tensor::from_slice(&[1.0], &[1], device).broadcast_to(&[n_affected])?,
            &denom,
        )?;
        let raw_alphas = client.div(&numer, &safe_denom)?;
        let zero_mask = client.lt(&client.abs(&denom)?, &eps_bcast)?;
        let zeros = Tensor::zeros(&[n_affected], DType::F64, device);
        let alphas = client.where_cond(&zero_mask, &zeros, &raw_alphas)?; // [n_affected]

        // Blend control points: new[i] = (1-alpha[i]) * cp[i-1] + alpha[i] * cp[i]
        // For the special case start==0: new[0] = cp[0] (alpha is irrelevant, we just keep it)
        let n_dims = cp.shape()[1];
        let alphas_col = alphas.reshape(&[n_affected, 1])?; // [n_affected, 1]
        let one_minus_alpha = client.sub(
            &Tensor::from_slice(&[1.0], &[1, 1], device).broadcast_to(&[n_affected, n_dims])?,
            &alphas_col.broadcast_to(&[n_affected, n_dims])?,
        )?;
        let alpha_broad = alphas_col.broadcast_to(&[n_affected, n_dims])?;

        // cp_curr = cp[start..start+n_affected]
        let cp_curr = cp.narrow(0, start, n_affected)?.contiguous();

        // cp_prev: for i>0, cp[i-1]; for i==0, cp[0] (doesn't matter, alpha masks it)
        let prev_start = if start > 0 { start - 1 } else { 0 };
        let cp_prev = if start > 0 {
            cp.narrow(0, prev_start, n_affected)?.contiguous()
        } else {
            // First row is cp[0] (placeholder), rest is cp[0..n_affected-1]
            let first = cp.narrow(0, 0, 1)?.contiguous();
            if n_affected > 1 {
                let rest = cp.narrow(0, 0, n_affected - 1)?.contiguous();
                client.cat(&[&first, &rest], 0)?
            } else {
                first
            }
        };

        let blended = client.add(
            &client.mul(&one_minus_alpha, &cp_prev)?,
            &client.mul(&alpha_broad, &cp_curr)?,
        )?;

        // If start == 0, the first blended point should just be cp[0]
        let blended = if start == 0 {
            let orig_first = cp.narrow(0, 0, 1)?.contiguous();
            if n_affected > 1 {
                let rest = blended.narrow(0, 1, n_affected - 1)?.contiguous();
                client.cat(&[&orig_first, &rest], 0)?
            } else {
                orig_first
            }
        } else {
            blended
        };

        // Assemble: [unchanged_before | blended | unchanged_after]
        let mut parts: Vec<Tensor<R>> = Vec::new();
        if start > 0 {
            parts.push(cp.narrow(0, 0, start)?.contiguous());
        }
        parts.push(blended);
        if current_span < n_cp {
            let remaining = n_cp - current_span;
            parts.push(cp.narrow(0, current_span, remaining)?.contiguous());
        }
        let refs: Vec<&Tensor<R>> = parts.iter().collect();
        cp = client.cat(&refs, 0)?;

        current_span += 1;
    }

    // Split at the fully-inserted knot
    let n_cp = cp.shape()[0];
    let split_idx = current_span - k + 1;

    let left_cp = cp.narrow(0, 0, split_idx + 1)?.contiguous();
    let right_cp = cp.narrow(0, split_idx, n_cp - split_idx)?.contiguous();

    let n_kn = knots.shape()[0];
    let left_end = split_idx + k + 1;
    let left_knots = knots.narrow(0, 0, left_end + 1)?.contiguous();
    let right_knots = knots.narrow(0, split_idx, n_kn - split_idx)?.contiguous();

    Ok((
        BSplineCurve {
            control_points: left_cp,
            knots: left_knots,
            degree: k,
        },
        BSplineCurve {
            control_points: right_cp,
            knots: right_knots,
            degree: k,
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
    fn test_bspline_curve_linear() {
        let (device, client) = setup();
        // Linear B-spline: 2 control points, degree 1, knots [0,0,1,1]
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[2, 2], &device);
        let knots = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let curve = BSplineCurve {
            control_points: cp,
            knots,
            degree: 1,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
        let result = bspline_curve_evaluate_impl(&client, &curve, &t).unwrap();
        let vals: Vec<f64> = result.to_vec();
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_curve_endpoints() {
        let (device, client) = setup();
        // Clamped cubic: 4 points, degree 3, knots [0,0,0,0,1,1,1,1]
        let cp = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 0.0],
            &[4, 2],
            &device,
        );
        let knots = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            &[8],
            &device,
        );
        let curve = BSplineCurve {
            control_points: cp,
            knots,
            degree: 3,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let result = bspline_curve_evaluate_impl(&client, &curve, &t).unwrap();
        let vals: Vec<f64> = result.to_vec();
        // Clamped: should pass through first and last control points
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
        assert!((vals[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_curve_derivative() {
        let (device, client) = setup();
        // Linear: derivative should be constant
        let cp = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 2.0, 4.0], &[2, 2], &device);
        let knots = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 1.0], &[4], &device);
        let curve = BSplineCurve {
            control_points: cp,
            knots,
            degree: 1,
        };
        let t = Tensor::<CpuRuntime>::from_slice(&[0.25, 0.75], &[2], &device);
        let deriv = bspline_curve_derivative_impl(&client, &curve, &t, 1).unwrap();
        let vals: Vec<f64> = deriv.to_vec();
        assert!((vals[0] - 2.0).abs() < 1e-8);
        assert!((vals[1] - 4.0).abs() < 1e-8);
        assert!((vals[2] - 2.0).abs() < 1e-8);
        assert!((vals[3] - 4.0).abs() < 1e-8);
    }
}
