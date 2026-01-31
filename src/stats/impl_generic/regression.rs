//! Generic regression implementations.
//!
//! This module provides Runtime-generic implementations of regression analysis.
//! All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Simple Linear Regression
//!
//! Fits the model y = β₀ + β₁x + ε using **Ordinary Least Squares (OLS)**.
//!
//! ## Formulas
//!
//! **Slope (β₁):**
//! ```text
//! β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
//!    = SS_xy / SS_xx
//! ```
//!
//! **Intercept (β₀):**
//! ```text
//! β₀ = ȳ - β₁x̄
//! ```
//!
//! **Correlation coefficient (r):**
//! ```text
//! r = SS_xy / √(SS_xx · SS_yy)
//! ```
//!
//! where:
//! - SS_xx = Σ(xᵢ - x̄)² (sum of squares of x deviations)
//! - SS_yy = Σ(yᵢ - ȳ)² (sum of squares of y deviations)
//! - SS_xy = Σ(xᵢ - x̄)(yᵢ - ȳ) (sum of cross-products)
//!
//! ## Standard Errors
//!
//! **Residual standard error:**
//! ```text
//! s² = SS_res / (n - 2)
//! SS_res = Σ(yᵢ - ŷᵢ)²
//! ```
//!
//! **Standard error of slope:**
//! ```text
//! SE(β₁) = √(s² / SS_xx)
//! ```
//!
//! **Standard error of intercept:**
//! ```text
//! SE(β₀) = SE(β₁) × √(1/n + x̄²/SS_xx)
//! ```
//!
//! ## P-Value
//!
//! Tests H₀: β₁ = 0 (no linear relationship) using:
//! ```text
//! t = β₁ / SE(β₁)
//! ```
//!
//! P-value is two-sided from Student's t-distribution with df = n - 2.
//!
//! ## Assumptions
//!
//! OLS assumes:
//! 1. **Linearity**: True relationship is linear
//! 2. **Independence**: Observations are independent
//! 3. **Homoscedasticity**: Constant variance of residuals
//! 4. **Normality**: Residuals are normally distributed (for valid p-values)
//!
//! Violations may require robust regression methods (not implemented here).

use crate::stats::helpers::extract_scalar;
use crate::stats::{ContinuousDistribution, LinregressResult, StudentT, validate_stats_dtype};
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of linear regression.
pub fn linregress_impl<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> Result<LinregressResult>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;
    validate_stats_dtype(y.dtype())?;

    if x.numel() != y.numel() {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "regression requires equal-length samples".to_string(),
        });
    }

    let n = x.numel();
    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "regression requires at least 2 samples".to_string(),
        });
    }

    let x_contig = x.contiguous();
    let y_contig = y.contiguous();

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();

    let mean_x = extract_scalar(&client.mean(&x_contig, &all_dims, false)?)?;
    let mean_y = extract_scalar(&client.mean(&y_contig, &all_dims, false)?)?;

    let mean_x_b = Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), mean_x, client.device());
    let mean_y_b = Tensor::<R>::full_scalar(y_contig.shape(), y.dtype(), mean_y, client.device());

    let dx = client.sub(&x_contig, &mean_x_b)?;
    let dy = client.sub(&y_contig, &mean_y_b)?;

    let dx_dy = client.mul(&dx, &dy)?;
    let dx_sq = client.mul(&dx, &dx)?;
    let dy_sq = client.mul(&dy, &dy)?;

    let ss_xy = extract_scalar(&client.sum(&dx_dy, &all_dims, false)?)?;
    let ss_xx = extract_scalar(&client.sum(&dx_sq, &all_dims, false)?)?;
    let ss_yy = extract_scalar(&client.sum(&dy_sq, &all_dims, false)?)?;

    if ss_xx == 0.0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "x has zero variance".to_string(),
        });
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let r = if ss_yy > 0.0 {
        ss_xy / (ss_xx * ss_yy).sqrt()
    } else {
        1.0
    };

    let n_f = n as f64;
    let df = n_f - 2.0;

    // Residual sum of squares
    let slope_tensor =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), slope, client.device());
    let y_pred_offset = client.mul(&x_contig, &slope_tensor)?;
    let intercept_tensor =
        Tensor::<R>::full_scalar(y_contig.shape(), y.dtype(), intercept, client.device());
    let y_pred = client.add(&y_pred_offset, &intercept_tensor)?;
    let residuals = client.sub(&y_contig, &y_pred)?;
    let residuals_sq = client.mul(&residuals, &residuals)?;
    let ss_res = extract_scalar(&client.sum(&residuals_sq, &all_dims, false)?)?;

    let mse = ss_res / df;
    let std_err = (mse / ss_xx).sqrt();

    // P-value for slope
    let t_stat = slope / std_err;
    let t_dist = StudentT::new(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create t distribution: {:?}", e),
    })?;

    let pvalue = 2.0 * t_dist.sf(t_stat.abs());
    let intercept_stderr = std_err * (1.0 / n_f + mean_x * mean_x / ss_xx).sqrt();

    Ok(LinregressResult {
        slope,
        intercept,
        rvalue: r,
        pvalue,
        stderr: std_err,
        intercept_stderr,
    })
}
