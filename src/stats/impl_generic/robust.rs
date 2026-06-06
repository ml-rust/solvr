//! Generic robust statistics implementations.
use crate::DType;

use crate::stats::helpers::{extract_scalar, tensor_median_scalar};
use crate::stats::traits::{RobustRegressionResult, validate_stats_dtype};
use numr::error::{Error, Result};
use numr::ops::{CompareOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of trimmed mean.
pub fn trim_mean_impl<R, C>(client: &C, x: &Tensor<R>, proportiontocut: f64) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    if !(0.0..0.5).contains(&proportiontocut) {
        return Err(Error::InvalidArgument {
            arg: "proportiontocut",
            reason: "must be in [0, 0.5)".to_string(),
        });
    }

    let x_contig = x.contiguous()?;
    let n = x_contig.numel();

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "trimmed mean requires at least 2 samples".to_string(),
        });
    }

    let sorted = client.sort(&x_contig, 0, false)?;
    let ncut = (n as f64 * proportiontocut).floor() as usize;

    if 2 * ncut >= n {
        return Err(Error::InvalidArgument {
            arg: "proportiontocut",
            reason: "proportion too large for sample size".to_string(),
        });
    }

    let trimmed = sorted.narrow(0, ncut, n - 2 * ncut)?.contiguous()?;
    let all_dims: Vec<usize> = (0..trimmed.ndim()).collect();
    client.mean(&trimmed, &all_dims, false)
}

/// Generic implementation of winsorized mean.
pub fn winsorized_mean_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    proportiontocut: f64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    if !(0.0..0.5).contains(&proportiontocut) {
        return Err(Error::InvalidArgument {
            arg: "proportiontocut",
            reason: "must be in [0, 0.5)".to_string(),
        });
    }

    let x_contig = x.contiguous()?;
    let n = x_contig.numel();

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "winsorized mean requires at least 2 samples".to_string(),
        });
    }

    let sorted = client.sort(&x_contig, 0, false)?;
    let ncut = (n as f64 * proportiontocut).floor() as usize;

    if 2 * ncut >= n {
        return Err(Error::InvalidArgument {
            arg: "proportiontocut",
            reason: "proportion too large for sample size".to_string(),
        });
    }

    // Single scalar transfer for boundary values (acceptable: convergence-like check)
    let low_val = extract_scalar(&sorted.narrow(0, ncut, 1)?)?;
    let high_val = extract_scalar(&sorted.narrow(0, n - ncut - 1, 1)?)?;

    let clamped = client.clamp(&x_contig, low_val, high_val)?;
    let all_dims: Vec<usize> = (0..clamped.ndim()).collect();
    client.mean(&clamped, &all_dims, false)
}

/// Generic implementation of median absolute deviation.
pub fn median_abs_deviation_impl<R, C>(client: &C, x: &Tensor<R>, scale: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous()?;
    let n = x_contig.numel();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "MAD requires at least 1 sample".to_string(),
        });
    }

    // Compute median (single scalar transfer)
    let median_val = tensor_median_scalar(client, &x_contig)?;

    // Compute |x - median| on device
    let median_t =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), median_val, client.device());
    let deviations = client.sub(&x_contig, &median_t)?;
    let abs_deviations = client.abs(&deviations)?;

    // Compute median of absolute deviations (single scalar transfer)
    let mad_val = tensor_median_scalar(client, &abs_deviations)?;

    let result = if scale { mad_val * 1.4826 } else { mad_val };
    Ok(Tensor::<R>::full_scalar(
        &[],
        x.dtype(),
        result,
        client.device(),
    ))
}

/// Generic implementation of Theil-Sen slope estimator.
///
/// Uses tensor broadcasting to compute all n*(n-1)/2 pairwise slopes in parallel.
pub fn theilslopes_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<RobustRegressionResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;
    validate_stats_dtype(y.dtype())?;

    let n = x.numel();
    if n != y.numel() {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "must have equal length".to_string(),
        });
    }
    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "Theil-Sen requires at least 2 points".to_string(),
        });
    }

    let device = client.device();
    let dtype = x.dtype();
    let x_contig = x.contiguous()?;
    let y_contig = y.contiguous()?;

    // Reshape to [n,1] and [1,n] for broadcasting → [n,n] pairwise diffs
    let x_col = x_contig.reshape(&[n, 1])?;
    let x_row = x_contig.reshape(&[1, n])?;
    let y_col = y_contig.reshape(&[n, 1])?;
    let y_row = y_contig.reshape(&[1, n])?;

    // dx[i,j] = x[j] - x[i], dy[i,j] = y[j] - y[i]
    let dx = client.sub(&x_row, &x_col)?;
    let dy = client.sub(&y_row, &y_col)?;

    // slopes[i,j] = dy[i,j] / dx[i,j]
    let slopes_matrix = client.div(&dy, &dx)?;

    // Create upper-triangle mask: i < j
    // Use arange to build row and col index tensors
    let row_idx = client.arange(0.0, n as f64, 1.0, dtype)?.reshape(&[n, 1])?;
    let col_idx = client.arange(0.0, n as f64, 1.0, dtype)?.reshape(&[1, n])?;
    let mask = client.lt(&row_idx, &col_idx)?;

    // Also mask out zero dx (identical x values): |dx| > eps
    let eps = Tensor::<R>::full_scalar(&[1, 1], dtype, 1e-15, device);
    let dx_abs = client.abs(&dx)?;
    let nonzero_mask = client.gt(&dx_abs, &eps)?;

    // Combined mask: upper triangle AND non-zero dx
    let valid_mask = client.mul(&mask, &nonzero_mask)?;

    // Extract valid slopes (masked_select requires U8 mask)
    let valid_mask_u8 = client.cast(&valid_mask, numr::dtype::DType::U8)?;
    let valid_slopes = client.masked_select(&slopes_matrix, &valid_mask_u8)?;

    if valid_slopes.numel() == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "all x values are identical".to_string(),
        });
    }

    // Median slope (single scalar transfer)
    let slope = tensor_median_scalar(client, &valid_slopes)?;

    // Intercepts: y - slope * x, then median (on device)
    let slope_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, slope, device);
    let slope_x = client.mul(&slope_t, &x_contig)?;
    let intercepts = client.sub(&y_contig, &slope_x)?;
    let intercept = tensor_median_scalar(client, &intercepts)?;

    // Confidence interval for slope (Conover 1980)
    let m = valid_slopes.numel();
    let sorted_slopes = client.sort(&valid_slopes, 0, false)?;
    let z = 1.96;
    let c =
        (z * (n as f64 * (n as f64 - 1.0) * (2.0 * n as f64 + 5.0) / 18.0).sqrt()).round() as usize;
    let low_slope = if c < m {
        extract_scalar(&sorted_slopes.narrow(0, c, 1)?)?
    } else {
        extract_scalar(&sorted_slopes.narrow(0, 0, 1)?)?
    };
    let high_slope = if m > c {
        extract_scalar(&sorted_slopes.narrow(0, m - 1 - c, 1)?)?
    } else {
        extract_scalar(&sorted_slopes.narrow(0, m - 1, 1)?)?
    };

    Ok(RobustRegressionResult {
        slope: Tensor::<R>::full_scalar(&[], dtype, slope, device),
        intercept: Tensor::<R>::full_scalar(&[], dtype, intercept, device),
        low_slope: Tensor::<R>::full_scalar(&[], dtype, low_slope, device),
        high_slope: Tensor::<R>::full_scalar(&[], dtype, high_slope, device),
    })
}

/// Generic implementation of Siegel repeated medians regression.
///
/// For each point i, computes median slope to all other points j, then takes
/// the overall median of those per-point medians. Uses tensor ops for the
/// pairwise slope computation and sorting.
pub fn siegelslopes_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<RobustRegressionResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;
    validate_stats_dtype(y.dtype())?;

    let n = x.numel();
    if n != y.numel() {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "must have equal length".to_string(),
        });
    }
    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "Siegel slopes requires at least 2 points".to_string(),
        });
    }

    let device = client.device();
    let dtype = x.dtype();
    let x_contig = x.contiguous()?;
    let y_contig = y.contiguous()?;

    // Build full n×n pairwise slope matrix via broadcasting
    let x_col = x_contig.reshape(&[n, 1])?;
    let x_row = x_contig.reshape(&[1, n])?;
    let y_col = y_contig.reshape(&[n, 1])?;
    let y_row = y_contig.reshape(&[1, n])?;

    let dx = client.sub(&x_row, &x_col)?; // [n, n]
    let dy = client.sub(&y_row, &y_col)?; // [n, n]

    // Replace zero dx with NaN to exclude from median
    let eps = Tensor::<R>::full_scalar(&[1, 1], dtype, 1e-15, device);
    let dx_abs = client.abs(&dx)?;
    let nonzero_mask = client.gt(&dx_abs, &eps)?;

    // Also mask diagonal (i == j)
    let row_idx = client.arange(0.0, n as f64, 1.0, dtype)?.reshape(&[n, 1])?;
    let col_idx = client.arange(0.0, n as f64, 1.0, dtype)?.reshape(&[1, n])?;
    let not_diag = client.ne(&row_idx, &col_idx)?;
    let valid_mask = client.mul(&nonzero_mask, &not_diag)?;

    let slopes_matrix = client.div(&dy, &dx)?; // [n, n], has inf/nan on diagonal

    // Replace invalid entries with +inf so they sort to the end
    let inf_val = Tensor::<R>::full_scalar(&[n, n], dtype, f64::INFINITY, device);
    let slopes_clean = client.where_cond(&valid_mask, &slopes_matrix, &inf_val)?;

    // Sort each row to get per-point sorted slopes: [n, n]
    let sorted_rows = client.sort(&slopes_clean, 1, false)?;

    // Count valid entries per row to find median index
    // valid_mask is bool; sum across dim=1 gives count per row
    // Cast bool to float for sum
    let ones = Tensor::<R>::full_scalar(&[n, n], dtype, 1.0, device);
    let zeros = Tensor::<R>::full_scalar(&[n, n], dtype, 0.0, device);
    let valid_float = client.where_cond(&valid_mask, &ones, &zeros)?;
    let counts = client.sum(&valid_float, &[1], false)?; // [n]

    // For each row i, median index = floor(count_i / 2)
    // Since all valid values sort before +inf, median is at sorted_rows[i, count_i/2]
    // We need to extract one element per row. Use gather.
    let two = Tensor::<R>::full_scalar(counts.shape(), dtype, 2.0, device);
    let median_indices_f = client.div(&counts, &two)?;
    // Floor and cast to I64 for gather
    let median_indices_floor = client.floor(&median_indices_f)?;
    let median_indices = client.cast(&median_indices_floor, numr::dtype::DType::I64)?;
    let median_indices_2d = median_indices.reshape(&[n, 1])?;

    // Gather median slope per row
    let per_point_medians = client.gather(&sorted_rows, 1, &median_indices_2d)?; // [n, 1]
    let per_point_medians_flat = per_point_medians.reshape(&[n])?;

    // Filter out rows where count was 0 (all x identical to point i)
    let zero_t = Tensor::<R>::full_scalar(counts.shape(), dtype, 0.0, device);
    let has_valid = client.gt(&counts, &zero_t)?;
    let has_valid_u8 = client.cast(&has_valid, numr::dtype::DType::U8)?;
    let valid_medians = client.masked_select(&per_point_medians_flat, &has_valid_u8)?;

    if valid_medians.numel() == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "all x values are identical".to_string(),
        });
    }

    // Overall slope = median of per-point medians
    let slope = tensor_median_scalar(client, &valid_medians)?;

    // Intercept = median of (y_i - slope * x_i)
    let slope_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, slope, device);
    let slope_x = client.mul(&slope_t, &x_contig)?;
    let intercepts = client.sub(&y_contig, &slope_x)?;
    let intercept = tensor_median_scalar(client, &intercepts)?;

    // Confidence interval (same approach as Theil-Sen)
    let ms = valid_medians.numel();
    let sorted_medians = client.sort(&valid_medians, 0, false)?;
    let z = 1.96;
    let c =
        (z * (n as f64 * (n as f64 - 1.0) * (2.0 * n as f64 + 5.0) / 18.0).sqrt()).round() as usize;
    let low_slope = if c < ms {
        extract_scalar(&sorted_medians.narrow(0, c, 1)?)?
    } else {
        extract_scalar(&sorted_medians.narrow(0, 0, 1)?)?
    };
    let high_slope = if ms > c {
        extract_scalar(&sorted_medians.narrow(0, ms - 1 - c, 1)?)?
    } else {
        extract_scalar(&sorted_medians.narrow(0, ms - 1, 1)?)?
    };

    Ok(RobustRegressionResult {
        slope: Tensor::<R>::full_scalar(&[], dtype, slope, device),
        intercept: Tensor::<R>::full_scalar(&[], dtype, intercept, device),
        low_slope: Tensor::<R>::full_scalar(&[], dtype, low_slope, device),
        high_slope: Tensor::<R>::full_scalar(&[], dtype, high_slope, device),
    })
}
