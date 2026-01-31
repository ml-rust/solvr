//! Generic descriptive statistics implementations.

use crate::stats::helpers::extract_scalar;
use crate::stats::{TensorDescriptiveStats, validate_stats_dtype};
use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of `describe` for any Runtime.
pub fn describe_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<TensorDescriptiveStats<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    if x.numel() == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "cannot compute statistics on empty tensor".to_string(),
        });
    }

    let x_contig = x.contiguous();
    let n = x_contig.numel();
    let n_f = n as f64;

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();

    let mean_tensor = client.mean(&x_contig, &all_dims, false)?;
    let var_tensor = client.var(&x_contig, &all_dims, false, 1)?;
    let min_tensor = client.min(&x_contig, &all_dims, false)?;
    let max_tensor = client.max(&x_contig, &all_dims, false)?;
    let std_tensor = client.std(&x_contig, &all_dims, false, 1)?;

    // For skewness and kurtosis, we need to compute moments
    let mean_val = extract_scalar(&mean_tensor)?;

    let mean_broadcast =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), mean_val, client.device());
    let centered = client.sub(&x_contig, &mean_broadcast)?;

    let centered_sq = client.mul(&centered, &centered)?;
    let centered_cu = client.mul(&centered_sq, &centered)?;
    let centered_qu = client.mul(&centered_sq, &centered_sq)?;

    let m2 = extract_scalar(&client.sum(&centered_sq, &all_dims, false)?)?;
    let m3 = extract_scalar(&client.sum(&centered_cu, &all_dims, false)?)?;
    let m4 = extract_scalar(&client.sum(&centered_qu, &all_dims, false)?)?;

    let skewness_val = if n > 2 && m2 > 0.0 {
        let m2_norm = m2 / n_f;
        (m3 / n_f) / m2_norm.powf(1.5)
    } else {
        0.0
    };

    let kurtosis_val = if n > 3 && m2 > 0.0 {
        let m2_norm = m2 / n_f;
        (m4 / n_f) / (m2_norm * m2_norm) - 3.0
    } else {
        0.0
    };

    let skewness_tensor = Tensor::<R>::full_scalar(&[], x.dtype(), skewness_val, client.device());
    let kurtosis_tensor = Tensor::<R>::full_scalar(&[], x.dtype(), kurtosis_val, client.device());

    Ok(TensorDescriptiveStats {
        nobs: n,
        min: min_tensor,
        max: max_tensor,
        mean: mean_tensor,
        variance: var_tensor,
        std: std_tensor,
        skewness: skewness_tensor,
        kurtosis: kurtosis_tensor,
    })
}

/// Generic implementation of `percentile` for any Runtime.
pub fn percentile_impl<R, C>(client: &C, x: &Tensor<R>, p: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    if !(0.0..=100.0).contains(&p) {
        return Err(Error::InvalidArgument {
            arg: "p",
            reason: format!("percentile must be in [0, 100], got {}", p),
        });
    }

    TensorOps::percentile(client, x, p, None, false)
}

/// Generic implementation of `iqr` for any Runtime.
pub fn iqr_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let q1 = percentile_impl(client, x, 25.0)?;
    let q3 = percentile_impl(client, x, 75.0)?;
    client.sub(&q3, &q1)
}

/// Generic implementation of `skewness` for any Runtime.
pub fn skewness_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
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
            reason: "skewness requires at least 3 samples".to_string(),
        });
    }

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();
    let mean_tensor = client.mean(&x_contig, &all_dims, false)?;
    let mean_val = extract_scalar(&mean_tensor)?;

    let mean_broadcast =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), mean_val, client.device());
    let centered = client.sub(&x_contig, &mean_broadcast)?;

    let centered_sq = client.mul(&centered, &centered)?;
    let centered_cu = client.mul(&centered_sq, &centered)?;

    let m2 = extract_scalar(&client.sum(&centered_sq, &all_dims, false)?)?;
    let m3 = extract_scalar(&client.sum(&centered_cu, &all_dims, false)?)?;

    let n_f = n as f64;
    let skew = if m2 > 0.0 {
        let m2_norm = m2 / n_f;
        (m3 / n_f) / m2_norm.powf(1.5)
    } else {
        0.0
    };

    Ok(Tensor::<R>::full_scalar(
        &[],
        x.dtype(),
        skew,
        client.device(),
    ))
}

/// Generic implementation of `kurtosis` for any Runtime.
pub fn kurtosis_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel();

    if n < 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "kurtosis requires at least 4 samples".to_string(),
        });
    }

    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();
    let mean_tensor = client.mean(&x_contig, &all_dims, false)?;
    let mean_val = extract_scalar(&mean_tensor)?;

    let mean_broadcast =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), mean_val, client.device());
    let centered = client.sub(&x_contig, &mean_broadcast)?;

    let centered_sq = client.mul(&centered, &centered)?;
    let centered_qu = client.mul(&centered_sq, &centered_sq)?;

    let m2 = extract_scalar(&client.sum(&centered_sq, &all_dims, false)?)?;
    let m4 = extract_scalar(&client.sum(&centered_qu, &all_dims, false)?)?;

    let n_f = n as f64;
    let kurt = if m2 > 0.0 {
        let m2_norm = m2 / n_f;
        (m4 / n_f) / (m2_norm * m2_norm) - 3.0
    } else {
        0.0
    };

    Ok(Tensor::<R>::full_scalar(
        &[],
        x.dtype(),
        kurt,
        client.device(),
    ))
}

/// Generic implementation of `zscore` for any Runtime.
pub fn zscore_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();

    let mean_val = extract_scalar(&client.mean(&x_contig, &all_dims, false)?)?;
    let std_val = extract_scalar(&client.std(&x_contig, &all_dims, false, 1)?)?;

    if std_val == 0.0 {
        return Ok(Tensor::<R>::zeros(
            x_contig.shape(),
            x.dtype(),
            client.device(),
        ));
    }

    let mean_broadcast =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), mean_val, client.device());
    let centered = client.sub(&x_contig, &mean_broadcast)?;

    let std_broadcast =
        Tensor::<R>::full_scalar(x_contig.shape(), x.dtype(), std_val, client.device());
    client.div(&centered, &std_broadcast)
}

/// Generic implementation of `sem` for any Runtime.
pub fn sem_impl<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    validate_stats_dtype(x.dtype())?;

    let x_contig = x.contiguous();
    let n = x_contig.numel() as f64;
    let all_dims: Vec<usize> = (0..x_contig.ndim()).collect();

    let std_val = extract_scalar(&client.std(&x_contig, &all_dims, false, 1)?)?;
    let sem_val = std_val / n.sqrt();

    Ok(Tensor::<R>::full_scalar(
        &[],
        x.dtype(),
        sem_val,
        client.device(),
    ))
}
