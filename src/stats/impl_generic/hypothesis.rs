//! Generic hypothesis testing implementations.
//!
//! This module provides Runtime-generic implementations of classical statistical
//! hypothesis tests. All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Implemented Tests
//!
//! ## T-Tests
//!
//! - **One-sample t-test** (`ttest_1samp_impl`): Tests if sample mean differs from
//!   a hypothesized population mean.
//!
//!   ```text
//!   t = (x̄ - μ₀) / (s / √n)
//!   ```
//!
//!   where x̄ is sample mean, μ₀ is hypothesized mean, s is sample std, n is sample size.
//!   P-value computed from Student's t-distribution with df = n - 1.
//!
//! - **Independent two-sample t-test** (`ttest_ind_impl`): Tests if two independent
//!   samples have different means. Uses **Welch's t-test** which does not assume
//!   equal variances.
//!
//!   ```text
//!   t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
//!   ```
//!
//!   Degrees of freedom computed via Welch-Satterthwaite approximation:
//!   ```text
//!   df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
//!   ```
//!
//! - **Paired t-test** (`ttest_rel_impl`): Tests if mean difference between paired
//!   observations is zero. Computed as one-sample t-test on differences.
//!
//! ## Correlation Tests
//!
//! - **Pearson correlation** (`pearsonr_impl`): Measures linear correlation between
//!   two variables. Returns correlation coefficient r ∈ [-1, 1].
//!
//!   ```text
//!   r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²]
//!   ```
//!
//!   P-value tests H₀: ρ = 0 using t-statistic: t = r√(n-2)/√(1-r²) with df = n - 2.
//!
//! - **Spearman rank correlation** (`spearmanr_impl`): Non-parametric measure of
//!   monotonic relationship. Computed as Pearson correlation on ranked data.
//!
//! # P-Value Computation
//!
//! All tests return two-sided p-values computed as:
//! ```text
//! p = 2 × P(T > |t|)  where T ~ t(df)
//! ```
//!
//! The Student's t-distribution CDF is computed via the regularized incomplete
//! beta function (see `StudentT::sf()` in continuous distributions).
//!
//! # Assumptions
//!
//! - **T-tests**: Assume approximately normal data or large sample size (CLT).
//!   Welch's test relaxes equal variance assumption.
//! - **Pearson**: Assumes linear relationship and bivariate normality for p-value.
//! - **Spearman**: Non-parametric, no distributional assumptions on data.

use crate::stats::helpers::{compute_ranks, extract_scalar};
use crate::stats::{ContinuousDistribution, StudentT, TensorTestResult, validate_stats_dtype};
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of one-sample t-test.
pub fn ttest_1samp_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    popmean: f64,
) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel();

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "t-test requires at least 2 samples".to_string(),
        });
    }

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();
    let mean_val = extract_scalar(&client.mean(&x_contig, &all_dims, false)?)?;
    let std_val = extract_scalar(&client.std(&x_contig, &all_dims, false, 1)?)?;

    let n_f = n as f64;
    let t_stat = (mean_val - popmean) / (std_val / n_f.sqrt());
    let df = n_f - 1.0;

    let t_dist = StudentT::new(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create t distribution: {:?}", e),
    })?;

    let pvalue = 2.0 * t_dist.sf(t_stat.abs());

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], x.dtype(), t_stat, client.device()),
        pvalue: Tensor::<R>::full_scalar(&[], x.dtype(), pvalue, client.device()),
    })
}

/// Generic implementation of independent two-sample t-test (Welch's).
pub fn ttest_ind_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(a.dtype())?;
    validate_stats_dtype(b.dtype())?;

    let a_contig = a.contiguous();
    let b_contig = b.contiguous();

    let n1 = a_contig.numel();
    let n2 = b_contig.numel();

    if n1 < 2 || n2 < 2 {
        return Err(Error::InvalidArgument {
            arg: "a/b",
            reason: "t-test requires at least 2 samples per group".to_string(),
        });
    }

    let all_dims_a: Vec<usize> = (0..a_contig.ndim()).collect();
    let all_dims_b: Vec<usize> = (0..b_contig.ndim()).collect();

    let mean1 = extract_scalar(&client.mean(&a_contig, &all_dims_a, false)?)?;
    let mean2 = extract_scalar(&client.mean(&b_contig, &all_dims_b, false)?)?;
    let var1 = extract_scalar(&client.var(&a_contig, &all_dims_a, false, 1)?)?;
    let var2 = extract_scalar(&client.var(&b_contig, &all_dims_b, false, 1)?)?;

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    // Welch's t-test (unequal variances)
    let se = (var1 / n1_f + var2 / n2_f).sqrt();
    let t_stat = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (var1 / n1_f + var2 / n2_f).powi(2);
    let denom = (var1 / n1_f).powi(2) / (n1_f - 1.0) + (var2 / n2_f).powi(2) / (n2_f - 1.0);
    let df = num / denom;

    let t_dist = StudentT::new(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create t distribution: {:?}", e),
    })?;

    let pvalue = 2.0 * t_dist.sf(t_stat.abs());

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], a.dtype(), t_stat, client.device()),
        pvalue: Tensor::<R>::full_scalar(&[], a.dtype(), pvalue, client.device()),
    })
}

/// Generic implementation of paired t-test.
pub fn ttest_rel_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(a.dtype())?;
    validate_stats_dtype(b.dtype())?;

    if a.numel() != b.numel() {
        return Err(Error::InvalidArgument {
            arg: "a/b",
            reason: "paired t-test requires equal-length samples".to_string(),
        });
    }

    let diff = client.sub(a, b)?;
    ttest_1samp_impl(client, &diff, 0.0)
}

/// Generic implementation of Pearson correlation.
pub fn pearsonr_impl<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;
    validate_stats_dtype(y.dtype())?;

    if x.numel() != y.numel() {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "correlation requires equal-length samples".to_string(),
        });
    }

    let n = x.numel();
    if n < 3 {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "correlation requires at least 3 samples".to_string(),
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

    let cov = extract_scalar(&client.sum(&dx_dy, &all_dims, false)?)?;
    let var_x = extract_scalar(&client.sum(&dx_sq, &all_dims, false)?)?;
    let var_y = extract_scalar(&client.sum(&dy_sq, &all_dims, false)?)?;

    let r = if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x * var_y).sqrt()
    } else {
        0.0
    };

    let n_f = n as f64;
    let t_stat = r * ((n_f - 2.0) / (1.0 - r * r)).sqrt();
    let df = n_f - 2.0;

    let t_dist = StudentT::new(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create t distribution: {:?}", e),
    })?;

    let pvalue = 2.0 * t_dist.sf(t_stat.abs());

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], x.dtype(), r, client.device()),
        pvalue: Tensor::<R>::full_scalar(&[], x.dtype(), pvalue, client.device()),
    })
}

/// Generic implementation of Spearman rank correlation.
pub fn spearmanr_impl<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;
    validate_stats_dtype(y.dtype())?;

    if x.numel() != y.numel() {
        return Err(Error::InvalidArgument {
            arg: "x/y",
            reason: "correlation requires equal-length samples".to_string(),
        });
    }

    let x_ranks = compute_ranks(client, x)?;
    let y_ranks = compute_ranks(client, y)?;

    pearsonr_impl(client, &x_ranks, &y_ranks)
}
