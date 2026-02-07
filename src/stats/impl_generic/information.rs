//! Generic information theory implementations.

use crate::stats::helpers::extract_scalar;
use crate::stats::validate_stats_dtype;
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of Shannon entropy.
pub fn entropy_impl<R, C>(client: &C, pk: &Tensor<R>, base: Option<f64>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(pk.dtype())?;

    let pk_contig = pk.contiguous();
    let n = pk_contig.numel();
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "pk",
            reason: "empty distribution".to_string(),
        });
    }

    // H = -Σ p * log(p), treating 0*log(0) = 0
    let epsilon = Tensor::<R>::full_scalar(pk_contig.shape(), pk.dtype(), 1e-300, client.device());
    let pk_safe = client.maximum(&pk_contig, &epsilon)?;
    let log_pk = client.log(&pk_safe)?;

    let p_log_p = client.mul(&pk_contig, &log_pk)?;

    let all_dims: Vec<usize> = (0..p_log_p.ndim()).collect();
    let sum = extract_scalar(&client.sum(&p_log_p, &all_dims, false)?)?;
    let mut h = -sum;

    if let Some(b) = base {
        h /= b.ln();
    }

    Ok(Tensor::<R>::full_scalar(
        &[],
        pk.dtype(),
        h,
        client.device(),
    ))
}

/// Generic implementation of differential entropy via k-NN spacing estimator.
///
/// For 1-D data, uses the sorted-spacing approach: for each point, the k-th
/// nearest neighbor distance is computed from the sorted array as
/// `max(x[i+k] - x[i], x[i] - x[i-k])`, avoiding O(n²) pairwise distances.
pub fn differential_entropy_impl<R, C>(client: &C, x: &Tensor<R>, k: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel();

    if n < k + 1 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("need at least {} samples for k={}", k + 1, k),
        });
    }
    if k == 0 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: "k must be at least 1".to_string(),
        });
    }

    let dtype = x.dtype();
    let device = client.device();

    // Sort data on device
    let sorted = client.sort(&x_contig, 0, false)?;

    // For 1-D sorted data, the k-th nearest neighbor of sorted[i] is:
    //   min distance from sorted[i-k..i-1] or sorted[i+1..i+k]
    // But for the KL estimator, we use: rho_i = distance to k-th NN
    // In sorted 1-D data, the k-th NN distance for sorted[i] is:
    //   Consider sorted[i-k] and sorted[i+k] (if they exist), pick the k-th closest
    //
    // Simplified approach: use spacing. For sorted data, compute
    //   forward spacing: sorted[i+k] - sorted[i] for i in [0, n-k)
    //   backward spacing: sorted[i] - sorted[i-k] for i in [k, n)
    // Then rho_i = min(forward[i], backward[i]) where applicable, but actually
    // for the Kozachenko-Leonenko estimator, we just need:
    //   rho_i = 2 * (distance to k-th NN)
    //
    // For simplicity and correctness, compute shifted differences on device:
    let head = sorted.narrow(0, 0, n - k)?; // sorted[0..n-k]
    let tail = sorted.narrow(0, k, n - k)?; // sorted[k..n]
    let spacings = client.sub(&tail, &head)?; // sorted[i+k] - sorted[i], length n-k

    // For the estimator we need log(2 * rho_i) for each point
    // Use the spacing as an approximation of 2*rho for interior points
    // (standard approach for 1-D KL estimator with sorted data)
    let epsilon = Tensor::<R>::full_scalar(spacings.shape(), dtype, 1e-300, device);
    let safe_spacings = client.maximum(&spacings, &epsilon)?;
    let log_spacings = client.log(&safe_spacings)?;

    let all_dims: Vec<usize> = (0..log_spacings.ndim()).collect();
    let log_sum = extract_scalar(&client.sum(&log_spacings, &all_dims, false)?)?;

    let n_eff = (n - k) as f64;
    let digamma_n = {
        use crate::stats::continuous::special::digamma;
        digamma(n as f64)
    };
    let digamma_k = {
        use crate::stats::continuous::special::digamma;
        digamma(k as f64)
    };

    let h = digamma_n - digamma_k + log_sum / n_eff;

    Ok(Tensor::<R>::full_scalar(&[], dtype, h, device))
}

/// Generic implementation of KL divergence.
pub fn kl_divergence_impl<R, C>(
    client: &C,
    pk: &Tensor<R>,
    qk: &Tensor<R>,
    base: Option<f64>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(pk.dtype())?;
    validate_stats_dtype(qk.dtype())?;

    if pk.numel() != qk.numel() {
        return Err(Error::InvalidArgument {
            arg: "pk/qk",
            reason: "distributions must have equal length".to_string(),
        });
    }

    let pk_contig = pk.contiguous();
    let qk_contig = qk.contiguous();

    // D_KL = Σ p * log(p/q) = Σ p * (log(p) - log(q))
    let epsilon = Tensor::<R>::full_scalar(pk_contig.shape(), pk.dtype(), 1e-300, client.device());
    let pk_safe = client.maximum(&pk_contig, &epsilon)?;
    let qk_safe = client.maximum(&qk_contig, &epsilon)?;

    let log_pk = client.log(&pk_safe)?;
    let log_qk = client.log(&qk_safe)?;
    let log_ratio = client.sub(&log_pk, &log_qk)?;

    let terms = client.mul(&pk_contig, &log_ratio)?;

    let all_dims: Vec<usize> = (0..terms.ndim()).collect();
    let sum = extract_scalar(&client.sum(&terms, &all_dims, false)?)?;

    let mut kl = sum;
    if let Some(b) = base {
        kl /= b.ln();
    }

    Ok(Tensor::<R>::full_scalar(
        &[],
        pk.dtype(),
        kl,
        client.device(),
    ))
}

/// Generic implementation of mutual information via histogram binning.
///
/// Computes bin indices and joint/marginal histograms entirely on device
/// using tensor operations.
pub fn mutual_information_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    bins: usize,
    base: Option<f64>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
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
    if n == 0 || bins == 0 {
        return Err(Error::InvalidArgument {
            arg: "bins",
            reason: "need positive bins and non-empty data".to_string(),
        });
    }

    let dtype = x.dtype();
    let device = client.device();
    let x_contig = x.contiguous();
    let y_contig = y.contiguous();

    // Compute min/max on device (single scalar transfers for range computation)
    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();
    let x_min = extract_scalar(&client.min(&x_contig, &all_dims, false)?)?;
    let x_max = extract_scalar(&client.max(&x_contig, &all_dims, false)?)?;
    let y_min = extract_scalar(&client.min(&y_contig, &all_dims, false)?)?;
    let y_max = extract_scalar(&client.max(&y_contig, &all_dims, false)?)?;

    let x_range = if (x_max - x_min).abs() < 1e-15 {
        1.0
    } else {
        x_max - x_min
    };
    let y_range = if (y_max - y_min).abs() < 1e-15 {
        1.0
    } else {
        y_max - y_min
    };

    // Compute bin indices on device: bin_i = clamp(round((x - min) / range * (bins-1)), 0, bins-1)
    let bins_f = (bins - 1) as f64;
    let x_min_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, x_min, device);
    let x_shifted = client.sub(&x_contig, &x_min_t)?;
    let x_scale_t = Tensor::<R>::full_scalar(x_contig.shape(), dtype, bins_f / x_range, device);
    let x_scaled = client.mul(&x_shifted, &x_scale_t)?;
    let x_rounded = client.round(&x_scaled)?;
    let x_bins = client.clamp(&x_rounded, 0.0, bins_f)?;

    let y_min_t = Tensor::<R>::full_scalar(y_contig.shape(), dtype, y_min, device);
    let y_shifted = client.sub(&y_contig, &y_min_t)?;
    let y_scale_t = Tensor::<R>::full_scalar(y_contig.shape(), dtype, bins_f / y_range, device);
    let y_scaled = client.mul(&y_shifted, &y_scale_t)?;
    let y_rounded = client.round(&y_scaled)?;
    let y_bins = client.clamp(&y_rounded, 0.0, bins_f)?;

    // Flatten to 1-D joint index: joint_idx = x_bin * bins + y_bin
    let bins_scale_t = Tensor::<R>::full_scalar(x_bins.shape(), dtype, bins as f64, device);
    let x_bins_scaled = client.mul(&x_bins, &bins_scale_t)?;
    let joint_idx_f = client.add(&x_bins_scaled, &y_bins)?;
    let joint_idx = client.cast(&joint_idx_f, numr::dtype::DType::I64)?;

    // Build joint histogram via scatter_reduce (sum ones at joint indices)
    let ones = Tensor::<R>::full_scalar(&[n], dtype, 1.0, device);
    let joint_zeros = Tensor::<R>::full_scalar(&[bins * bins], dtype, 0.0, device);
    let joint_hist = client.scatter_reduce(
        &joint_zeros,
        0,
        &joint_idx,
        &ones,
        numr::ops::ScatterReduceOp::Sum,
        true,
    )?;

    // Normalize to joint probability
    let n_t = Tensor::<R>::full_scalar(&[1], dtype, n as f64, device);
    let pxy = client.div(&joint_hist, &n_t)?; // [bins*bins]

    // Marginals: reshape to [bins, bins], sum along axes
    let pxy_2d = pxy.reshape(&[bins, bins])?;
    let px = client.sum(&pxy_2d, &[1], false)?; // [bins]
    let py = client.sum(&pxy_2d, &[0], false)?; // [bins]

    // I(X;Y) = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))
    // Compute outer product p(x) * p(y) → [bins, bins]
    let px_col = px.reshape(&[bins, 1])?;
    let py_row = py.reshape(&[1, bins])?;
    let px_py = client.mul(&px_col, &py_row)?; // [bins, bins]
    let px_py_flat = px_py.reshape(&[bins * bins])?;

    // Compute log(pxy / (px*py)) where both > 0
    let epsilon = Tensor::<R>::full_scalar(&[bins * bins], dtype, 1e-300, device);
    let pxy_safe = client.maximum(&pxy, &epsilon)?;
    let pxpy_safe = client.maximum(&px_py_flat, &epsilon)?;
    let ratio = client.div(&pxy_safe, &pxpy_safe)?;
    let log_ratio = client.log(&ratio)?;

    // p(x,y) * log(p(x,y) / (p(x)*p(y)))
    let terms = client.mul(&pxy, &log_ratio)?;

    let all_dims_joint: Vec<usize> = (0..terms.ndim()).collect();
    let mut mi = extract_scalar(&client.sum(&terms, &all_dims_joint, false)?)?;

    if let Some(b) = base {
        mi /= b.ln();
    }

    // MI can be slightly negative due to floating point; clamp to 0
    mi = mi.max(0.0);

    Ok(Tensor::<R>::full_scalar(&[], dtype, mi, device))
}
