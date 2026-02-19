//! B-spline interpolation generic implementation (fully on-device).
//!
//! Implements B-spline construction via collocation matrix, evaluation via
//! batched basis function computation, derivatives via knot differencing,
//! and integration via anti-derivative construction.
//!
//! All operations use tensor ops — zero GPU↔CPU transfers in algorithm code.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use crate::interpolate::traits::bspline::{BSpline, BSplineBoundary};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, ScalarOps};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

// ============ Public API ============

/// Construct an interpolating B-spline from data points (fully on-device).
pub fn make_interp_spline_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    degree: usize,
    boundary: &BSplineBoundary,
) -> InterpolateResult<BSpline<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n = x.shape()[0];

    if n < 2 {
        return Err(InterpolateError::InsufficientData {
            required: 2,
            actual: n,
            context: "make_interp_spline".to_string(),
        });
    }
    if degree == 0 {
        return Err(InterpolateError::InvalidParameter {
            parameter: "degree".to_string(),
            message: "degree must be >= 1".to_string(),
        });
    }
    if n <= degree {
        return Err(InterpolateError::InsufficientData {
            required: degree + 1,
            actual: n,
            context: "make_interp_spline: need at least degree+1 points".to_string(),
        });
    }

    // Monotonicity check (on-device, single scalar extraction for control flow)
    let dx = client.sub(
        &x.narrow(0, 1, n - 1)?.contiguous(),
        &x.narrow(0, 0, n - 1)?.contiguous(),
    )?;
    let zero_dx = Tensor::zeros(&[n - 1], DType::F64, device);
    let non_pos = client.le(&dx, &zero_dx)?;
    let bad_count = client.sum(&non_pos, &[0], false)?;
    if bad_count.item::<f64>().unwrap_or(0.0) > 0.0 {
        return Err(InterpolateError::NotMonotonic {
            context: "make_interp_spline".to_string(),
        });
    }

    // Build knot vector on-device
    let knots = build_knot_vector_tensor(client, x, degree, boundary, n)?;
    let n_knots = knots.shape()[0];
    let n_coeffs = n_knots - degree - 1;

    // Build collocation matrix and rhs based on boundary
    let (col_mat, rhs_col) =
        build_collocation_tensor(client, x, y, &knots, degree, boundary, n, n_coeffs)?;

    // Solve the collocation system on-device
    let coeffs_col = LinearAlgebraAlgorithms::solve(client, &col_mat, &rhs_col).map_err(|e| {
        InterpolateError::NumericalError {
            message: format!("Failed to solve B-spline collocation system: {}", e),
        }
    })?;
    let coefficients = coeffs_col.reshape(&[n_coeffs])?;

    Ok(BSpline {
        knots,
        coefficients,
        degree,
    })
}

/// Evaluate a B-spline at new points (fully on-device).
///
/// Uses batched basis function evaluation via tensor broadcasting and
/// the Cox-de Boor recurrence, then matmul with coefficients.
pub fn bspline_evaluate_impl<R, C>(
    client: &C,
    spline: &BSpline<R>,
    x_new: &Tensor<R>,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let m = x_new.shape()[0];
    let n_knots = spline.knots.shape()[0];
    let n_coeffs = n_knots - spline.degree - 1;

    // Compute basis matrix [m, n_coeffs] on-device
    let basis = compute_basis_matrix(client, x_new, &spline.knots, spline.degree, n_coeffs)?;

    // result = basis @ coefficients
    let coeffs_col = spline.coefficients.reshape(&[n_coeffs, 1])?;
    let result_col = client.matmul(&basis, &coeffs_col)?;
    Ok(result_col.reshape(&[m])?)
}

/// Evaluate B-spline derivative at new points (fully on-device).
///
/// Differentiates the B-spline coefficients using the knot differencing formula,
/// then evaluates the resulting lower-degree spline.
pub fn bspline_derivative_impl<R, C>(
    client: &C,
    spline: &BSpline<R>,
    x_new: &Tensor<R>,
    order: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if order == 0 {
        return bspline_evaluate_impl(client, spline, x_new);
    }
    if order > spline.degree {
        let m = x_new.shape()[0];
        let device = client.device();
        return Ok(Tensor::zeros(&[m], DType::F64, device));
    }

    // Differentiate the B-spline representation on-device
    let deriv_spline = differentiate_bspline_tensor(client, spline, order)?;
    bspline_evaluate_impl(client, &deriv_spline, x_new)
}

/// Compute the definite integral of a B-spline over [a, b] (fully on-device).
///
/// Constructs the anti-derivative B-spline using cumsum, then evaluates at endpoints.
pub fn bspline_integrate_impl<R, C>(
    client: &C,
    spline: &BSpline<R>,
    a: f64,
    b: f64,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let k = spline.degree;
    let n_coeffs = spline.coefficients.shape()[0];
    let n_knots = spline.knots.shape()[0];

    // Anti-derivative: C_i = sum_{j=0}^{i-1} c_j * (t_{j+k+1} - t_j) / (k+1)
    let t_hi = spline.knots.narrow(0, k + 1, n_coeffs)?.contiguous();
    let t_lo = spline.knots.narrow(0, 0, n_coeffs)?.contiguous();
    let dt = client.sub(&t_hi, &t_lo)?;
    let terms = client.mul_scalar(
        &client.mul(&spline.coefficients, &dt)?,
        1.0 / (k + 1) as f64,
    )?;

    // C_0 = 0, C_i = cumsum(terms)[i-1]
    let cumsum = client.cumsum(&terms, 0)?;
    let zero_1 = Tensor::zeros(&[1], DType::F64, device);
    let anti_coeffs = client.cat(&[&zero_1, &cumsum], 0)?;

    // Anti-derivative knot vector: add one copy at each end
    let first = spline.knots.narrow(0, 0, 1)?.contiguous();
    let last = spline.knots.narrow(0, n_knots - 1, 1)?.contiguous();
    let anti_knots = client.cat(&[&first, &spline.knots, &last], 0)?;

    let anti_spline = BSpline {
        knots: anti_knots,
        coefficients: anti_coeffs,
        degree: k + 1,
    };

    // Evaluate anti-derivative at b and a, return difference
    let ab = Tensor::from_slice(&[b, a], &[2], device);
    let vals = bspline_evaluate_impl(client, &anti_spline, &ab)?;
    let val_b = vals.narrow(0, 0, 1)?.contiguous();
    let val_a = vals.narrow(0, 1, 1)?.contiguous();
    Ok(client.sub(&val_b, &val_a)?)
}

// ============ Core computation: batched basis matrix ============

/// Compute the B-spline basis matrix [m, n_coeffs] on-device.
///
/// Uses the Cox-de Boor recurrence with batched tensor operations:
/// 1. Degree-0 basis via element-wise comparison (broadcasting)
/// 2. Recurrence for degrees 1..k via narrow + broadcast + safe division
pub(crate) fn compute_basis_matrix<R, C>(
    client: &C,
    x: &Tensor<R>,
    knots: &Tensor<R>,
    degree: usize,
    n_coeffs: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let m = x.shape()[0];
    let n_knots = knots.shape()[0];
    let n_basis_0 = n_knots - 1; // number of degree-0 basis functions

    // x_col: [m, 1] for broadcasting
    let x_col = x.reshape(&[m, 1])?.contiguous();

    // === Degree-0 basis: B0[i,j] = 1 if knots[j] <= x[i] < knots[j+1] ===
    let knots_left = knots
        .narrow(0, 0, n_basis_0)?
        .contiguous()
        .reshape(&[1, n_basis_0])?;
    let knots_right = knots
        .narrow(0, 1, n_basis_0)?
        .contiguous()
        .reshape(&[1, n_basis_0])?;

    let x_broad = x_col.broadcast_to(&[m, n_basis_0])?.contiguous();
    let kl_broad = knots_left.broadcast_to(&[m, n_basis_0])?.contiguous();
    let kr_broad = knots_right.broadcast_to(&[m, n_basis_0])?.contiguous();

    let ge_left = client.ge(&x_broad, &kl_broad)?;
    let lt_right = client.lt(&x_broad, &kr_broad)?;
    let in_span = client.mul(&ge_left, &lt_right)?;

    // Handle right endpoint: x == knots[last] → set last valid span basis to 1
    let right_span_idx = n_knots - degree - 2;
    let idx = client.arange(0.0, n_basis_0 as f64, 1.0, DType::F64)?;
    let target = Tensor::full_scalar(&[n_basis_0], DType::F64, right_span_idx as f64, device);
    let right_col_mask = client.eq(&idx, &target)?; // [n_basis_0]

    let last_knot = knots
        .narrow(0, n_knots - 1, 1)?
        .contiguous()
        .reshape(&[1])?;
    let at_right = client.ge(x, &last_knot.broadcast_to(&[m])?.contiguous())?; // [m]

    let right_correction = client.mul(
        &at_right
            .reshape(&[m, 1])?
            .broadcast_to(&[m, n_basis_0])?
            .contiguous(),
        &right_col_mask
            .reshape(&[1, n_basis_0])?
            .broadcast_to(&[m, n_basis_0])?
            .contiguous(),
    )?;

    let mut basis = client.maximum(&in_span, &right_correction)?;

    // === Recurrence for degrees 1..k ===
    let eps_val = Tensor::full_scalar(&[1], DType::F64, 1e-300, device);

    for p in 1..=degree {
        let n_active = n_knots - p - 1;

        let basis_left = basis.narrow(1, 0, n_active)?.contiguous();
        let basis_right = basis.narrow(1, 1, n_active)?.contiguous();

        // w1[j] = (x - knots[j]) / (knots[j+p] - knots[j])
        let kj = knots
            .narrow(0, 0, n_active)?
            .contiguous()
            .reshape(&[1, n_active])?;
        let kjp = knots
            .narrow(0, p, n_active)?
            .contiguous()
            .reshape(&[1, n_active])?;
        let denom1 = client.sub(&kjp, &kj)?;
        let numer1 = client.sub(
            &x_col.broadcast_to(&[m, n_active])?.contiguous(),
            &kj.broadcast_to(&[m, n_active])?.contiguous(),
        )?;

        let abs_d1 = client.abs(&denom1)?;
        let eps_broad1 = eps_val.broadcast_to(&[1, n_active])?.contiguous();
        let d1_safe = client.maximum(&abs_d1, &eps_broad1)?;
        let zero_1n = Tensor::zeros(&[1, n_active], DType::F64, device);
        let mask1 = client.gt(&abs_d1, &zero_1n)?;
        let w1 = client.mul(
            &client.div(&numer1, &d1_safe.broadcast_to(&[m, n_active])?.contiguous())?,
            &mask1.broadcast_to(&[m, n_active])?.contiguous(),
        )?;

        // w2[j] = (knots[j+p+1] - x) / (knots[j+p+1] - knots[j+1])
        let kj1 = knots
            .narrow(0, 1, n_active)?
            .contiguous()
            .reshape(&[1, n_active])?;
        let kjp1 = knots
            .narrow(0, p + 1, n_active)?
            .contiguous()
            .reshape(&[1, n_active])?;
        let denom2 = client.sub(&kjp1, &kj1)?;
        let numer2 = client.sub(
            &kjp1.broadcast_to(&[m, n_active])?.contiguous(),
            &x_col.broadcast_to(&[m, n_active])?.contiguous(),
        )?;

        let abs_d2 = client.abs(&denom2)?;
        let d2_safe = client.maximum(&abs_d2, &eps_broad1)?;
        let mask2 = client.gt(&abs_d2, &zero_1n)?;
        let w2 = client.mul(
            &client.div(&numer2, &d2_safe.broadcast_to(&[m, n_active])?.contiguous())?,
            &mask2.broadcast_to(&[m, n_active])?.contiguous(),
        )?;

        let term1 = client.mul(&w1, &basis_left)?;
        let term2 = client.mul(&w2, &basis_right)?;
        let new_active = client.add(&term1, &term2)?;

        // Pad to n_basis_0 columns for next iteration
        if n_active < n_basis_0 {
            let padding = Tensor::zeros(&[m, n_basis_0 - n_active], DType::F64, device);
            basis = client.cat(&[&new_active, &padding], 1)?;
        } else {
            basis = new_active;
        }
    }

    // Extract first n_coeffs columns
    Ok(basis.narrow(1, 0, n_coeffs)?.contiguous())
}

// ============ Knot vector construction (on-device) ============

/// Build the B-spline knot vector on-device.
pub(crate) fn build_knot_vector_tensor<R, C>(
    client: &C,
    x: &Tensor<R>,
    degree: usize,
    boundary: &BSplineBoundary,
    n: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let k = degree;
    let x_first = x.narrow(0, 0, 1)?.contiguous();
    let x_last = x.narrow(0, n - 1, 1)?.contiguous();
    let first_rep = x_first.broadcast_to(&[k + 1])?.contiguous();
    let last_rep = x_last.broadcast_to(&[k + 1])?.contiguous();

    let interior = match boundary {
        BSplineBoundary::NotAKnot => {
            // Schoenberg-Whitney averaging: t_j = (x[j+1] + ... + x[j+k]) / k
            let n_interior = n.saturating_sub(k + 1);
            if n_interior == 0 {
                None
            } else {
                let mut knot_sum = x.narrow(0, 1, n_interior)?.contiguous();
                for offset in 1..k {
                    let shifted = x.narrow(0, 1 + offset, n_interior)?.contiguous();
                    knot_sum = client.add(&knot_sum, &shifted)?;
                }
                Some(client.mul_scalar(&knot_sum, 1.0 / k as f64)?)
            }
        }
        BSplineBoundary::Clamped { .. } | BSplineBoundary::Natural => {
            if n > 2 {
                Some(x.narrow(0, 1, n - 2)?.contiguous())
            } else {
                None
            }
        }
    };

    let knots = match interior {
        Some(int) => client.cat(&[&first_rep, &int, &last_rep], 0)?,
        None => client.cat(&[&first_rep, &last_rep], 0)?,
    };

    Ok(knots)
}

// ============ Collocation system (on-device) ============

#[allow(clippy::too_many_arguments)]
/// Build the collocation matrix and rhs for the B-spline system (on-device).
fn build_collocation_tensor<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    knots: &Tensor<R>,
    degree: usize,
    boundary: &BSplineBoundary,
    n: usize,
    n_coeffs: usize,
) -> InterpolateResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let device = client.device();

    match boundary {
        BSplineBoundary::NotAKnot => {
            // Simple collocation: basis_matrix @ coeffs = y
            let col_mat = compute_basis_matrix(client, x, knots, degree, n_coeffs)?;
            let rhs_col = y.reshape(&[n, 1])?;
            Ok((col_mat, rhs_col))
        }
        BSplineBoundary::Clamped { left, right } => {
            // n_rows = n + 2: derivative at x[0], interpolation at x[0..n], derivative at x[n-1]
            let n_rows = n + 2;

            // Derivative basis at x[0]
            let x_left = x.narrow(0, 0, 1)?.contiguous();
            let dbasis_left = compute_deriv_basis(client, &x_left, knots, degree, n_coeffs)?;

            // Interpolation basis at all x
            let basis = compute_basis_matrix(client, x, knots, degree, n_coeffs)?;

            // Derivative basis at x[n-1]
            let x_right = x.narrow(0, n - 1, 1)?.contiguous();
            let dbasis_right = compute_deriv_basis(client, &x_right, knots, degree, n_coeffs)?;

            // Stack: [dbasis_left; basis; dbasis_right] → [n+2, n_coeffs]
            let col_mat = client.cat(&[&dbasis_left, &basis, &dbasis_right], 0)?;

            // RHS: [left, y[0], ..., y[n-1], right]
            let left_t = Tensor::from_slice(&[*left], &[1, 1], device);
            let right_t = Tensor::from_slice(&[*right], &[1, 1], device);
            let y_col = y.reshape(&[n, 1])?;
            let rhs_col = client.cat(&[&left_t, &y_col, &right_t], 0)?;

            debug_assert_eq!(col_mat.shape()[0], n_rows);
            debug_assert_eq!(rhs_col.shape()[0], n_rows);

            Ok((col_mat, rhs_col))
        }
        BSplineBoundary::Natural => {
            // n_rows = n + 2: 2nd deriv=0 at x[0], interpolation, 2nd deriv=0 at x[n-1]
            let n_rows = n + 2;

            let x_left = x.narrow(0, 0, 1)?.contiguous();
            let d2basis_left = compute_deriv2_basis(client, &x_left, knots, degree, n_coeffs)?;

            let basis = compute_basis_matrix(client, x, knots, degree, n_coeffs)?;

            let x_right = x.narrow(0, n - 1, 1)?.contiguous();
            let d2basis_right = compute_deriv2_basis(client, &x_right, knots, degree, n_coeffs)?;

            let col_mat = client.cat(&[&d2basis_left, &basis, &d2basis_right], 0)?;

            let zero_1 = Tensor::zeros(&[1, 1], DType::F64, device);
            let y_col = y.reshape(&[n, 1])?;
            let rhs_col = client.cat(&[&zero_1, &y_col, &zero_1], 0)?;

            debug_assert_eq!(col_mat.shape()[0], n_rows);
            debug_assert_eq!(rhs_col.shape()[0], n_rows);

            Ok((col_mat, rhs_col))
        }
    }
}

// ============ Derivative basis computation (on-device) ============

/// Compute 1st derivative of B-spline basis functions at given points.
///
/// dB_{i,k}(t) = k * [B_{i,k-1}(t)/(t_{i+k}-t_i) - B_{i+1,k-1}(t)/(t_{i+k+1}-t_{i+1})]
fn compute_deriv_basis<R, C>(
    client: &C,
    x: &Tensor<R>,
    knots: &Tensor<R>,
    degree: usize,
    n_coeffs: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if degree == 0 {
        let m = x.shape()[0];
        let device = client.device();
        return Ok(Tensor::zeros(&[m, n_coeffs], DType::F64, device));
    }

    let device = client.device();
    let m = x.shape()[0];
    let n_lower = n_coeffs + 1; // degree-(k-1) basis functions

    // Evaluate degree-(k-1) basis: [m, n_lower]
    let lower = compute_basis_matrix(client, x, knots, degree - 1, n_lower)?;

    let lower_left = lower.narrow(1, 0, n_coeffs)?.contiguous(); // [m, n_coeffs]
    let lower_right = lower.narrow(1, 1, n_coeffs)?.contiguous(); // [m, n_coeffs]

    // Denominators from knot differences
    let t_lo = knots
        .narrow(0, 0, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let t_hi_k = knots
        .narrow(0, degree, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let denom1 = client.sub(&t_hi_k, &t_lo)?;

    let t_lo1 = knots
        .narrow(0, 1, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let t_hi_k1 = knots
        .narrow(0, degree + 1, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let denom2 = client.sub(&t_hi_k1, &t_lo1)?;

    // Safe division with zero masking
    let eps = Tensor::full_scalar(&[1, n_coeffs], DType::F64, 1e-300, device);
    let zero = Tensor::zeros(&[1, n_coeffs], DType::F64, device);

    let abs_d1 = client.abs(&denom1)?;
    let d1_safe = client.maximum(&abs_d1, &eps)?;
    let mask1 = client.gt(&abs_d1, &zero)?;
    let term1 = client.mul(
        &client.div(
            &lower_left,
            &d1_safe.broadcast_to(&[m, n_coeffs])?.contiguous(),
        )?,
        &mask1.broadcast_to(&[m, n_coeffs])?.contiguous(),
    )?;

    let abs_d2 = client.abs(&denom2)?;
    let d2_safe = client.maximum(&abs_d2, &eps)?;
    let mask2 = client.gt(&abs_d2, &zero)?;
    let term2 = client.mul(
        &client.div(
            &lower_right,
            &d2_safe.broadcast_to(&[m, n_coeffs])?.contiguous(),
        )?,
        &mask2.broadcast_to(&[m, n_coeffs])?.contiguous(),
    )?;

    let deriv = client.mul_scalar(&client.sub(&term1, &term2)?, degree as f64)?;
    Ok(deriv)
}

/// Compute 2nd derivative of B-spline basis functions at given points.
///
/// Applies the derivative formula twice: first get dB of degree-(k-1),
/// then apply the formula again.
fn compute_deriv2_basis<R, C>(
    client: &C,
    x: &Tensor<R>,
    knots: &Tensor<R>,
    degree: usize,
    n_coeffs: usize,
) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if degree < 2 {
        let m = x.shape()[0];
        let device = client.device();
        return Ok(Tensor::zeros(&[m, n_coeffs], DType::F64, device));
    }

    let device = client.device();
    let m = x.shape()[0];
    let n_lower = n_coeffs + 1;

    // Get derivative of degree-(k-1) basis: [m, n_lower]
    let d1_lower = compute_deriv_basis(client, x, knots, degree - 1, n_lower)?;

    let d1_left = d1_lower.narrow(1, 0, n_coeffs)?.contiguous();
    let d1_right = d1_lower.narrow(1, 1, n_coeffs)?.contiguous();

    let t_lo = knots
        .narrow(0, 0, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let t_hi_k = knots
        .narrow(0, degree, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let denom1 = client.sub(&t_hi_k, &t_lo)?;

    let t_lo1 = knots
        .narrow(0, 1, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let t_hi_k1 = knots
        .narrow(0, degree + 1, n_coeffs)?
        .contiguous()
        .reshape(&[1, n_coeffs])?;
    let denom2 = client.sub(&t_hi_k1, &t_lo1)?;

    let eps = Tensor::full_scalar(&[1, n_coeffs], DType::F64, 1e-300, device);
    let zero = Tensor::zeros(&[1, n_coeffs], DType::F64, device);

    let abs_d1 = client.abs(&denom1)?;
    let d1_safe = client.maximum(&abs_d1, &eps)?;
    let mask1 = client.gt(&abs_d1, &zero)?;
    let term1 = client.mul(
        &client.div(
            &d1_left,
            &d1_safe.broadcast_to(&[m, n_coeffs])?.contiguous(),
        )?,
        &mask1.broadcast_to(&[m, n_coeffs])?.contiguous(),
    )?;

    let abs_d2 = client.abs(&denom2)?;
    let d2_safe = client.maximum(&abs_d2, &eps)?;
    let mask2 = client.gt(&abs_d2, &zero)?;
    let term2 = client.mul(
        &client.div(
            &d1_right,
            &d2_safe.broadcast_to(&[m, n_coeffs])?.contiguous(),
        )?,
        &mask2.broadcast_to(&[m, n_coeffs])?.contiguous(),
    )?;

    let deriv2 = client.mul_scalar(&client.sub(&term1, &term2)?, degree as f64)?;
    Ok(deriv2)
}

// ============ Coefficient differentiation (on-device) ============

/// Differentiate a B-spline representation by `order` times (on-device).
///
/// c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
/// New knots: remove first and last knot each time.
pub(crate) fn differentiate_bspline_tensor<R, C>(
    client: &C,
    spline: &BSpline<R>,
    order: usize,
) -> InterpolateResult<BSpline<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let mut current_knots = spline.knots.clone();
    let mut current_coeffs = spline.coefficients.clone();
    let mut current_degree = spline.degree;

    for _ in 0..order {
        if current_degree == 0 {
            break;
        }
        let n = current_coeffs.shape()[0];
        let k = current_degree;
        let n_knots = current_knots.shape()[0];

        // c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
        let c_hi = current_coeffs.narrow(0, 1, n - 1)?.contiguous();
        let c_lo = current_coeffs.narrow(0, 0, n - 1)?.contiguous();
        let dc = client.sub(&c_hi, &c_lo)?;

        let t_hi = current_knots.narrow(0, k + 1, n - 1)?.contiguous();
        let t_lo = current_knots.narrow(0, 1, n - 1)?.contiguous();
        let dt = client.sub(&t_hi, &t_lo)?;

        // Safe division (zero dt → zero coefficient)
        let eps = Tensor::full_scalar(&[n - 1], DType::F64, 1e-300, client.device());
        let abs_dt = client.abs(&dt)?;
        let dt_safe = client.maximum(&abs_dt, &eps)?;
        let zero = Tensor::zeros(&[n - 1], DType::F64, client.device());
        let mask = client.gt(&abs_dt, &zero)?;
        let new_coeffs =
            client.mul_scalar(&client.mul(&client.div(&dc, &dt_safe)?, &mask)?, k as f64)?;

        // Remove first and last knot
        let new_knots = current_knots.narrow(0, 1, n_knots - 2)?.contiguous();

        current_coeffs = new_coeffs;
        current_knots = new_knots;
        current_degree -= 1;
    }

    Ok(BSpline {
        knots: current_knots,
        coefficients: current_coeffs,
        degree: current_degree,
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
    fn test_linear_bspline() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let spline = make_interp_spline_impl(&client, &x, &y, 1, &BSplineBoundary::NotAKnot)
            .expect("linear bspline failed");

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let result = bspline_evaluate_impl(&client, &spline, &x_new).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for (i, &v) in vals.iter().enumerate() {
            let expected = 1.0 + 2.0 * i as f64;
            assert!(
                (v - expected).abs() < 1e-10,
                "point {}: {} vs {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_cubic_bspline_interpolates_knots() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0], &[5], &device);

        let spline = make_interp_spline_impl(&client, &x, &y, 3, &BSplineBoundary::NotAKnot)
            .expect("cubic bspline failed");

        let result = bspline_evaluate_impl(&client, &spline, &x).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for (i, &v) in vals.iter().enumerate() {
            let expected = (i * i) as f64;
            assert!(
                (v - expected).abs() < 1e-8,
                "knot {}: {} vs {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_bspline_quadratic() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);

        let spline = make_interp_spline_impl(&client, &x, &y, 2, &BSplineBoundary::NotAKnot)
            .expect("quadratic bspline failed");

        let result = bspline_evaluate_impl(&client, &spline, &x).unwrap();
        let vals: Vec<f64> = result.to_vec();

        for (i, &v) in vals.iter().enumerate() {
            let expected = (i * i) as f64;
            assert!(
                (v - expected).abs() < 1e-6,
                "knot {}: {} vs {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_bspline_derivative_linear() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let spline = make_interp_spline_impl(&client, &x, &y, 1, &BSplineBoundary::NotAKnot)
            .expect("linear bspline for deriv");

        let x_new = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
        let deriv = bspline_derivative_impl(&client, &spline, &x_new, 1).unwrap();
        let vals: Vec<f64> = deriv.to_vec();

        for (i, &v) in vals.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-8, "deriv at {}: {} vs 2.0", i, v);
        }
    }

    #[test]
    fn test_bspline_integrate_linear() {
        let (device, client) = setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0, 6.0], &[4], &device);

        let spline = make_interp_spline_impl(&client, &x, &y, 1, &BSplineBoundary::NotAKnot)
            .expect("linear bspline for integrate");

        let result = bspline_integrate_impl(&client, &spline, 0.0, 3.0).unwrap();
        let val: Vec<f64> = result.to_vec();

        // Integral of 2x from 0 to 3 = 9
        assert!((val[0] - 9.0).abs() < 1e-6, "integral: {} vs 9.0", val[0]);
    }
}
