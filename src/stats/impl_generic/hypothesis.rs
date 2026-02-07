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
use crate::stats::traits::LeveneCenter;
use crate::stats::{
    ChiSquared, ContinuousDistribution, FDistribution, Normal, StudentT, TensorTestResult,
    validate_stats_dtype,
};
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

/// Generic implementation of one-way ANOVA (F-test).
pub fn f_oneway_impl<R, C>(client: &C, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    if groups.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "ANOVA requires at least 2 groups".to_string(),
        });
    }

    for g in groups {
        validate_stats_dtype(g.dtype())?;
        if g.numel() < 1 {
            return Err(Error::InvalidArgument {
                arg: "groups",
                reason: "each group must have at least 1 sample".to_string(),
            });
        }
    }

    let k = groups.len() as f64;
    let dtype = groups[0].dtype();
    let device = client.device();

    // Compute group means and sizes
    let mut group_means = Vec::new();
    let mut group_sizes = Vec::new();
    let mut n_total = 0usize;
    let mut grand_sum = 0.0;

    for g in groups {
        let g_contig = g.contiguous();
        let ni = g_contig.numel();
        let all_dims: Vec<usize> = (0..g_contig.ndim()).collect();
        let mean_i = extract_scalar(&client.mean(&g_contig, &all_dims, false)?)?;
        grand_sum += mean_i * ni as f64;
        group_means.push(mean_i);
        group_sizes.push(ni);
        n_total += ni;
    }

    let grand_mean = grand_sum / n_total as f64;

    // Between-group sum of squares: SSB = Σ n_i * (mean_i - grand_mean)^2
    let mut ssb = 0.0;
    for i in 0..groups.len() {
        ssb += group_sizes[i] as f64 * (group_means[i] - grand_mean).powi(2);
    }

    // Within-group sum of squares: SSW = ΣΣ (x_ij - mean_i)^2
    let mut ssw = 0.0;
    for (i, g) in groups.iter().enumerate() {
        let g_contig = g.contiguous();
        let all_dims: Vec<usize> = (0..g_contig.ndim()).collect();
        let var_i = extract_scalar(&client.var(&g_contig, &all_dims, false, 0)?)?;
        ssw += var_i * group_sizes[i] as f64;
    }

    let df_between = k - 1.0;
    let df_within = n_total as f64 - k;

    if df_within <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "insufficient total samples for ANOVA".to_string(),
        });
    }

    let msb = ssb / df_between;
    let msw = ssw / df_within;

    let f_stat = if msw > 0.0 { msb / msw } else { f64::INFINITY };

    let f_dist = FDistribution::new(df_between, df_within).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create F distribution: {:?}", e),
    })?;
    let pvalue = f_dist.sf(f_stat);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, f_stat, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of Kruskal-Wallis H-test.
pub fn kruskal_impl<R, C>(client: &C, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    if groups.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "Kruskal-Wallis requires at least 2 groups".to_string(),
        });
    }

    for g in groups {
        validate_stats_dtype(g.dtype())?;
        if g.numel() < 1 {
            return Err(Error::InvalidArgument {
                arg: "groups",
                reason: "each group must have at least 1 sample".to_string(),
            });
        }
    }

    let dtype = groups[0].dtype();
    let device = client.device();

    // Combine all groups on device and compute ranks
    let group_contigs: Vec<Tensor<R>> = groups.iter().map(|g| g.contiguous()).collect();
    let group_refs: Vec<&Tensor<R>> = group_contigs.iter().collect();
    let group_sizes: Vec<usize> = groups.iter().map(|g| g.numel()).collect();

    let combined = client.cat(&group_refs, 0)?;
    let n_total = combined.numel();
    let ranks = compute_ranks(client, &combined)?;

    // H = [12 / (N(N+1))] * Σ (R_i^2 / n_i) - 3(N+1)
    // Compute rank sums per group using narrow on device
    let n_f = n_total as f64;
    let mut offset = 0;
    let mut sum_term = 0.0;

    for &ni in &group_sizes {
        let group_ranks = ranks.narrow(0, offset, ni)?;
        let all_dims: Vec<usize> = (0..group_ranks.ndim()).collect();
        let rank_sum = extract_scalar(&client.sum(&group_ranks, &all_dims, false)?)?;
        sum_term += rank_sum * rank_sum / ni as f64;
        offset += ni;
    }

    let h = 12.0 / (n_f * (n_f + 1.0)) * sum_term - 3.0 * (n_f + 1.0);
    let df = groups.len() as f64 - 1.0;

    let chi2 = ChiSquared::new_f64(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create chi-squared distribution: {:?}", e),
    })?;
    let pvalue = chi2.sf(h);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, h, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of Friedman chi-squared test.
pub fn friedmanchisquare_impl<R, C>(
    client: &C,
    groups: &[&Tensor<R>],
) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    if groups.len() < 3 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "Friedman test requires at least 3 groups".to_string(),
        });
    }

    let n = groups[0].numel();
    for g in groups {
        validate_stats_dtype(g.dtype())?;
        if g.numel() != n {
            return Err(Error::InvalidArgument {
                arg: "groups",
                reason: "all groups must have equal length (repeated measures)".to_string(),
            });
        }
    }

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "need at least 2 subjects".to_string(),
        });
    }

    let dtype = groups[0].dtype();
    let device = client.device();
    let k = groups.len(); // number of treatments

    // Stack groups into [k, n] matrix, then rank each column (subject)
    // by sorting along dim=0 and using scatter to assign ranks
    let mut group_contigs = Vec::new();
    for g in groups {
        let c = g.contiguous();
        group_contigs.push(c.reshape(&[1, n])?);
    }
    let group_refs: Vec<&Tensor<R>> = group_contigs.iter().collect();
    let stacked = client.cat(&group_refs, 0)?; // [k, n]

    // argsort along dim=0 gives rank ordering per subject
    let indices = client.argsort(&stacked, 0, false)?; // [k, n]

    // Build ranks: scatter 1-based ranks into original positions per column
    let ranks_seq = {
        // Create [k, 1] rank values 1..k, broadcast to [k, n]
        let r = client.arange(1.0, (k + 1) as f64, 1.0, dtype)?;
        let r_col = r.reshape(&[k, 1])?;
        r_col.broadcast_to(&[k, n])?
    };
    let zeros = Tensor::<R>::full_scalar(&[k, n], dtype, 0.0, device);
    let ranks = client.scatter(&zeros, 0, &indices, &ranks_seq)?; // [k, n]

    // Rank sums per treatment: sum along dim=1 → [k]
    let rank_sums = client.sum(&ranks, &[1], false)?; // [k]

    // χ² = [12 / (n*k*(k+1))] * Σ R_j^2 - 3*n*(k+1)
    let n_f = n as f64;
    let k_f = k as f64;

    let rank_sums_sq = client.mul(&rank_sums, &rank_sums)?;
    let all_dims_k: Vec<usize> = (0..rank_sums_sq.ndim()).collect();
    let sum_r_sq = extract_scalar(&client.sum(&rank_sums_sq, &all_dims_k, false)?)?;
    let chi2_stat = 12.0 / (n_f * k_f * (k_f + 1.0)) * sum_r_sq - 3.0 * n_f * (k_f + 1.0);

    let df = k_f - 1.0;
    let chi2_dist = ChiSquared::new_f64(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create chi-squared distribution: {:?}", e),
    })?;
    let pvalue = chi2_dist.sf(chi2_stat);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, chi2_stat, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of Shapiro-Wilk test.
pub fn shapiro_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel();

    if n < 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Shapiro-Wilk requires at least 3 samples".to_string(),
        });
    }

    let dtype = x.dtype();
    let device = client.device();

    // Sort data on device
    let sorted = client.sort(&x_contig, 0, false)?;

    // Compute expected normal order statistics using Blom's approximation
    // This is a small coefficient vector computed from ppf (CPU-side is acceptable
    // since ppf is not a tensor op — these are algorithm coefficients, not data)
    let std_normal = Normal::standard();
    let mut m = vec![0.0f64; n];
    for i in 0..n {
        let p = (i as f64 + 1.0 - 0.375) / (n as f64 + 0.25);
        m[i] = std_normal.ppf(p).unwrap_or(0.0);
    }

    // Normalize: a = m / ||m||
    let m_norm: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    let a: Vec<f64> = m.iter().map(|v| v / m_norm).collect();

    // Transfer coefficients to device tensor for dot product
    let a_tensor = Tensor::<R>::from_slice(&a, &[n], device);

    // W = (Σ a_i * x_(i))^2 / Σ (x_i - x_bar)^2
    // Numerator: dot product a · sorted on device
    let a_sorted = client.mul(&a_tensor, &sorted)?;
    let all_dims: Vec<usize> = (0..a_sorted.ndim()).collect();
    let numerator = extract_scalar(&client.sum(&a_sorted, &all_dims, false)?)?;
    let numerator_sq = numerator * numerator;

    // Denominator: SS on device
    let mean_val = extract_scalar(&client.mean(&x_contig, &all_dims, false)?)?;
    let mean_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, mean_val, device);
    let dx = client.sub(&x_contig, &mean_t)?;
    let dx2 = client.mul(&dx, &dx)?;
    let ss = extract_scalar(&client.sum(&dx2, &all_dims, false)?)?;

    let w = if ss > 0.0 { numerator_sq / ss } else { 1.0 };

    // Approximate p-value using normal transformation of W
    // Using Royston's approximation for p-value
    let n_f = n as f64;
    let ln_n = n_f.ln();
    let mu = -1.2725 + 1.0521 * ln_n;
    let sigma = (1.0308 - 0.26758 * ln_n).exp();
    let z = ((1.0 - w).ln() - mu) / sigma;

    let normal = Normal::standard();
    let pvalue = normal.sf(z);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, w, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of D'Agostino-Pearson normaltest.
pub fn normaltest_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel();

    if n < 20 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "normaltest requires at least 20 samples".to_string(),
        });
    }

    let dtype = x.dtype();
    let device = client.device();
    let n_f = n as f64;

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();
    let mean_val = extract_scalar(&client.mean(&x_contig, &all_dims, false)?)?;

    // Compute central moments
    let mean_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, mean_val, device);
    let dx = client.sub(&x_contig, &mean_t)?;
    let dx2 = client.mul(&dx, &dx)?;
    let dx3 = client.mul(&dx2, &dx)?;
    let dx4 = client.mul(&dx3, &dx)?;

    let m2 = extract_scalar(&client.sum(&dx2, &all_dims, false)?)? / n_f;
    let m3 = extract_scalar(&client.sum(&dx3, &all_dims, false)?)? / n_f;
    let m4 = extract_scalar(&client.sum(&dx4, &all_dims, false)?)? / n_f;

    // Sample skewness: g1 = m3 / m2^(3/2)
    let g1 = m3 / m2.powf(1.5);
    // Sample kurtosis: g2 = m4 / m2^2 - 3
    let g2 = m4 / (m2 * m2) - 3.0;

    // D'Agostino's z-score for skewness
    let y = g1 * ((n_f + 1.0) * (n_f + 3.0) / (6.0 * (n_f - 2.0))).sqrt();
    let beta2 = 3.0 * (n_f * n_f + 27.0 * n_f - 70.0) * (n_f + 1.0) * (n_f + 3.0)
        / ((n_f - 2.0) * (n_f + 5.0) * (n_f + 7.0) * (n_f + 9.0));
    let w2 = (2.0 * (beta2 - 1.0)).sqrt() - 1.0;
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let z_s = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();

    // Anscombe-Glynn z-score for kurtosis
    let mean_k = 3.0 * (n_f - 1.0) / (n_f + 1.0) - 3.0; // expected excess kurtosis
    let var_k = 24.0 * n_f * (n_f - 2.0) * (n_f - 3.0)
        / ((n_f + 1.0) * (n_f + 1.0) * (n_f + 3.0) * (n_f + 5.0));
    let z_k = (g2 - mean_k) / var_k.sqrt();

    // Omnibus statistic: K^2 = z_s^2 + z_k^2 ~ χ²(2)
    let k2 = z_s * z_s + z_k * z_k;

    let chi2 = ChiSquared::new_f64(2.0).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create chi-squared distribution: {:?}", e),
    })?;
    let pvalue = chi2.sf(k2);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, k2, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of Levene's test.
pub fn levene_impl<R, C>(
    client: &C,
    groups: &[&Tensor<R>],
    center: LeveneCenter,
) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    if groups.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "Levene's test requires at least 2 groups".to_string(),
        });
    }

    for g in groups {
        validate_stats_dtype(g.dtype())?;
        if g.numel() < 2 {
            return Err(Error::InvalidArgument {
                arg: "groups",
                reason: "each group must have at least 2 samples".to_string(),
            });
        }
    }

    let dtype = groups[0].dtype();
    let device = client.device();
    let k = groups.len();

    // Compute center values and absolute deviations for each group on device
    use crate::stats::helpers::tensor_median_scalar;

    let mut deviation_tensors: Vec<Tensor<R>> = Vec::new();
    let mut n_total = 0usize;

    for g in groups {
        let g_contig = g.contiguous();
        let ni = g_contig.numel();
        n_total += ni;
        let all_dims: Vec<usize> = (0..g_contig.ndim()).collect();

        // Compute center value (single scalar transfer per group — acceptable)
        let center_val = match center {
            LeveneCenter::Mean => extract_scalar(&client.mean(&g_contig, &all_dims, false)?)?,
            LeveneCenter::Median => tensor_median_scalar(client, &g_contig)?,
            LeveneCenter::TrimmedMean => {
                let sorted = client.sort(&g_contig, 0, false)?;
                let ncut = (ni as f64 * 0.1).floor() as usize;
                let trimmed = sorted.narrow(0, ncut, ni - 2 * ncut)?;
                let t_dims: Vec<usize> = (0..trimmed.ndim()).collect();
                extract_scalar(&client.mean(&trimmed, &t_dims, false)?)?
            }
        };

        // |x - center| on device
        let center_t = Tensor::<R>::full_scalar(g_contig.shape(), dtype, center_val, device);
        let diff = client.sub(&g_contig, &center_t)?;
        let abs_dev = client.abs(&diff)?;
        deviation_tensors.push(abs_dev);
    }

    // Run one-way ANOVA on absolute deviations using tensor ops
    let dev_refs: Vec<&Tensor<R>> = deviation_tensors.iter().collect();
    let all_devs = client.cat(&dev_refs, 0)?; // [n_total]
    let all_dims_total: Vec<usize> = (0..all_devs.ndim()).collect();
    let grand_mean = extract_scalar(&client.mean(&all_devs, &all_dims_total, false)?)?;

    let mut ssb = 0.0;
    let mut ssw = 0.0;

    for dev_t in &deviation_tensors {
        let ni = dev_t.numel() as f64;
        let dims: Vec<usize> = (0..dev_t.ndim()).collect();
        let group_mean = extract_scalar(&client.mean(dev_t, &dims, false)?)?;
        ssb += ni * (group_mean - grand_mean).powi(2);
        let var_i = extract_scalar(&client.var(dev_t, &dims, false, 0)?)?;
        ssw += var_i * ni;
    }

    let df1 = k as f64 - 1.0;
    let df2 = n_total as f64 - k as f64;
    let f_stat = if ssw > 0.0 {
        (ssb / df1) / (ssw / df2)
    } else {
        f64::INFINITY
    };

    let f_dist = FDistribution::new(df1, df2).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create F distribution: {:?}", e),
    })?;
    let pvalue = f_dist.sf(f_stat);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, f_stat, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}

/// Generic implementation of Bartlett's test.
pub fn bartlett_impl<R, C>(client: &C, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    if groups.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "groups",
            reason: "Bartlett's test requires at least 2 groups".to_string(),
        });
    }

    for g in groups {
        validate_stats_dtype(g.dtype())?;
        if g.numel() < 2 {
            return Err(Error::InvalidArgument {
                arg: "groups",
                reason: "each group must have at least 2 samples".to_string(),
            });
        }
    }

    let dtype = groups[0].dtype();
    let device = client.device();
    let k = groups.len();

    let mut vars = Vec::new();
    let mut sizes = Vec::new();

    for g in groups {
        let g_contig = g.contiguous();
        let ni = g_contig.numel();
        let all_dims: Vec<usize> = (0..g_contig.ndim()).collect();
        let var_i = extract_scalar(&client.var(&g_contig, &all_dims, false, 1)?)?;
        vars.push(var_i);
        sizes.push(ni);
    }

    // Pooled variance
    let mut sp2_num = 0.0;
    let mut sp2_den = 0.0;
    for i in 0..k {
        let df_i = (sizes[i] - 1) as f64;
        sp2_num += df_i * vars[i];
        sp2_den += df_i;
    }
    let sp2 = sp2_num / sp2_den;

    if sp2 <= 0.0 {
        return Ok(TensorTestResult {
            statistic: Tensor::<R>::full_scalar(&[], dtype, 0.0, device),
            pvalue: Tensor::<R>::full_scalar(&[], dtype, 1.0, device),
        });
    }

    // Bartlett's statistic
    let mut sum_log = 0.0;
    let mut sum_inv = 0.0;
    for i in 0..k {
        let df_i = (sizes[i] - 1) as f64;
        sum_log += df_i * vars[i].ln();
        sum_inv += 1.0 / df_i;
    }

    let n_minus_k = sp2_den; // Σ(n_i - 1)
    let numerator = n_minus_k * sp2.ln() - sum_log;
    let correction = 1.0 + (1.0 / (3.0 * (k as f64 - 1.0))) * (sum_inv - 1.0 / n_minus_k);

    let t_stat = numerator / correction;
    let df = k as f64 - 1.0;

    let chi2 = ChiSquared::new_f64(df).map_err(|e| Error::InvalidArgument {
        arg: "df",
        reason: format!("failed to create chi-squared distribution: {:?}", e),
    })?;
    let pvalue = chi2.sf(t_stat);

    Ok(TensorTestResult {
        statistic: Tensor::<R>::full_scalar(&[], dtype, t_stat, device),
        pvalue: Tensor::<R>::full_scalar(&[], dtype, pvalue, device),
    })
}
